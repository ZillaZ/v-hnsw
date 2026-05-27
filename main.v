module hnsw

import os
import math
import rand
import encoding.binary
import arrays

// ------------------------------------------------------------
// Node
// ------------------------------------------------------------

struct Node[T] {
	token u64
	layer u16
mut:
	value T
	// neighbours is a flat list; per-layer slices are tracked via
	// layer_neighbour_counts so that we can correctly cap connections
	// per layer without mixing them.
	neighbours             []u64
	layer_neighbour_counts []int // len == layer+1; sum == neighbours.len
}

// to_bytes serialises the fixed-size node header into one slice and
// the padded 100-slot edge table into a second slice.
// Layout of the header slice:
//   [token: 8][layer: 2][value: T.byte_size()][neighbours_len: 8]
// Layout of the edge slice:
//   [slot0: 8][slot1: 8]...[slot99: 8]   (always 800 bytes)
@[manualfree]
fn (self Node[T]) to_bytes() ([]u8, []u8) {
	mut ret := []u8{len: 0, cap: int(Node.byte_size[T]())}
	token_b := binary.big_endian_get_u64(self.token)
	layer_b := binary.big_endian_get_u16(self.layer)
	value_b := self.value.to_bytes()
	nbrs_len_b := binary.big_endian_get_u64(u64(self.neighbours.len))
	// Edge record: 100 slots of 8 bytes each (800 bytes total).
	// Layout: [neighbour tokens ... | zeroed padding ... | layer_neighbour_counts]
	// layer_neighbour_counts occupies the LAST (layer+1) slots, stored highest index last.
	// This is safe as long as neighbours.len + layer_neighbour_counts.len <= 100.
	mut edge_slots := [][]u8{len: 100, init: binary.big_endian_get_u64(u64(0))}
	for i, nb in self.neighbours {
		edge_slots[i] = binary.big_endian_get_u64(nb)
	}
	// Write counts into the tail slots.
	counts_len := self.layer_neighbour_counts.len // == layer + 1
	for i, cnt in self.layer_neighbour_counts {
		tail_idx := 100 - counts_len + i
		edge_slots[tail_idx] = binary.big_endian_get_u64(u64(cnt))
	}

	ret << token_b
	ret << layer_b
	ret << value_b
	ret << nbrs_len_b
	edge_flat := arrays.flatten(edge_slots)
	unsafe {
		edge_slots.free()
		token_b.free()
		layer_b.free()
		value_b.free()
		nbrs_len_b.free()
	}
	return ret, edge_flat
}

// byte_size returns the size of the *header* portion only (used for
// seeking in db.nodes).  Edge data lives in the separate db.edges file.
fn Node.byte_size[T]() u64 {
	return 8 + 2 + T.byte_size() + 8
}

// from_bytes reconstructs a Node from the merged header+edge bytes.
// Caller is responsible for passing arrays.merge(header_buf, edge_buf).
fn Node.from_bytes[T](bytes []u8) Node[T] {
	min_size := Node.byte_size[T]() + 100 * 8
	if u64(bytes.len) < min_size {
		panic('byte vector is too small (${bytes.len} vs ${min_size})')
	}

	token := binary.big_endian_u64_at(bytes, 0)
	layer := binary.big_endian_u16_at(bytes, 8)
	value := T.from_bytes(bytes[10..])

	size := u64(T.byte_size())
	neighbours_len := binary.big_endian_u64_at(bytes, int(10 + size))
	// Edge data starts right after the header.
	edge_base := int(Node.byte_size[T]())

	mut neighbours := []u64{cap: int(neighbours_len)}
	for i in u64(0) .. neighbours_len {
		idx := binary.big_endian_u64_at(bytes, edge_base + int(8 * i))
		neighbours << idx
	}

	// Restore layer_neighbour_counts from the tail slots of the edge record.
	counts_len := int(layer) + 1
	mut layer_neighbour_counts := []int{len: 10, init: 0}
	for i in 0 .. counts_len {
		tail_idx := 100 - counts_len + i
		layer_neighbour_counts[i] = int(binary.big_endian_u64_at(bytes, edge_base + tail_idx * 8))
	}

	ret := Node[T]{
		token:                  token
		layer:                  layer
		value:                  value
		neighbours:             neighbours
		layer_neighbour_counts: layer_neighbour_counts
	}
	return ret
}

@[unsafe]
fn (node &Node[T]) free() {
	unsafe {
		node.neighbours.free()
		node.layer_neighbour_counts.free()
	}
}

// ------------------------------------------------------------
// HNSW
// ------------------------------------------------------------

pub struct HNSW[T] {
mut:
	file                 os.File
	edges                os.File
	has_entry_point      bool
	entry_point          u64
	top_layer            u16
	node_count           u64
	max_neighbours       int
	max_neighbours0      int
	ef_construction      int
	normalization_factor f64
}

pub fn new_hnsw[T](max_neighbours int, ef_construction int) HNSW[T] {
	// Use w+b to create files with read+write access (os.create opens wb, which is write-only).
	file := os.open_file('db.nodes', 'w+b') or { panic(err) }
	edges := os.open_file('db.edges', 'w+b') or { panic(err) }
	m := max_neighbours
	return HNSW[T]{
		file:                 file
		edges:                edges
		max_neighbours:       m
		max_neighbours0:      2 * m
		ef_construction:      ef_construction
		normalization_factor: 1.0 / math.log(f64(m))
	}
}

// ------------------------------------------------------------
// Disk I/O helpers
// ------------------------------------------------------------
@[manualfree]
fn (mut self HNSW[T]) write_node(node Node[T]) {
	mut node_bytes, mut edge_bytes := node.to_bytes()
	self.file.seek(i64(node.token) * i64(Node.byte_size[T]()), os.SeekMode.start) or { panic(err) }
	written := self.file.write(node_bytes) or { panic(err) }
	if written != node_bytes.len {
		panic('short write on node header for token ${node.token}: ${written} vs ${node_bytes.len}')
	}
	self.edges.seek(i64(node.token) * 100 * 8, os.SeekMode.start) or { panic(err) }
	ewritten := self.edges.write(edge_bytes) or { panic(err) }
	if ewritten != edge_bytes.len {
		panic('short write on node edges for token ${node.token}: ${ewritten} vs ${edge_bytes.len}')
	}
	self.file.flush()
	self.edges.flush()
	unsafe {
		node_bytes.free()
		edge_bytes.free()
	}
}

@[manualfree]
fn (mut self HNSW[T]) read_value(index u64) !Node[T] {
	node_size := i64(Node.byte_size[T]())
	mut header_buf := []u8{len: int(Node.byte_size[T]())}
	nread := self.file.read_bytes_into(u64(index) * u64(node_size), mut header_buf) or { return err }
	if nread != header_buf.len {
		unsafe { header_buf.free() }
		return error('short read on header for token ${index}: ${nread} vs ${header_buf.len}')
	}
	mut edge_buf := []u8{len: 100 * 8}
	nedge := self.edges.read_bytes_into(u64(index) * 100 * 8, mut edge_buf) or { return err }
	if nedge != edge_buf.len {
		unsafe { header_buf.free() }
		unsafe { edge_buf.free() }
		return error('short read on edges for token ${index}: ${nedge} vs ${edge_buf.len}')
	}
	header_buf << edge_buf
	ret := Node.from_bytes[T](header_buf)
	unsafe {
		header_buf.free()
		edge_buf.free()
	}
	return ret
}

fn (mut self HNSW[T]) alloc_node(value T, layer u16) Node[T] {
	token := self.node_count
	self.node_count += 1
	ret := Node[T]{
		token:                  token
		layer:                  layer
		value:                  value
		layer_neighbour_counts: []int{len: 10, init: 0}
	}
	return ret
}

// ------------------------------------------------------------
// Distance / neighbour selection
// ------------------------------------------------------------

fn (self HNSW[T]) nearest(query T, candidates []u64) u64 {
	mut best := candidates[0]
	mut best_dist := f64(math.max_f64)
	for c in candidates {
		node := unsafe { self.read_value(c) or { continue } }
		d := query.distance_to(node.value)
		if d < best_dist {
			best_dist = d
			best = c
		}
	}
	return best
}

fn (mut self HNSW[T]) get_layer_neighbours(node Node[T], layer u16) []u64 {
	if int(layer) >= node.layer_neighbour_counts.len {
		return []
	}
	// neighbours is flat, ordered highest layer first.
	// offset = sum of counts for layers strictly above `layer`.
	mut offset := 0
	for l := node.layer_neighbour_counts.len - 1; l > int(layer); l-- {
		offset += node.layer_neighbour_counts[l]
	}
	count := node.layer_neighbour_counts[int(layer)]
	return node.neighbours[offset..math.min[int](offset + count, node.neighbours.len)]
}

struct OrdHelper[T] {
	distance f64
	value    u64
}

fn (a OrdHelper[T]) < (b OrdHelper[T]) bool {
	return a.distance < b.distance
}

struct OrdHelperMax[T] {
	distance f64
	value    u64
}

fn (a OrdHelperMax[T]) < (b OrdHelperMax[T]) bool {
	return a.distance > b.distance
}

// Custom min-heap that exposes .free() for manual memory management.
struct MinHeap[T] {
mut:
	data []T
}

fn (mut heap MinHeap[T]) insert(item T) {
	heap.data << item
	mut child := heap.data.len - 1
	mut parent := (child - 1) / 2
	for heap.data[parent] > heap.data[child] {
		heap.data[parent], heap.data[child] = heap.data[child], heap.data[parent]
		child = parent
		parent = (child - 1) / 2
		if child <= 0 { break }
	}
}

fn (mut heap MinHeap[T]) pop() !T {
	if heap.data.len == 0 {
		return error('Heap is empty')
	} else if heap.data.len == 1 {
		return heap.data.pop()
	}
	item := heap.data[0]
	heap.data[0] = heap.data.pop()
	mut parent := 0
	for {
		left := 2 * parent + 1
		right := 2 * parent + 2
		mut smallest := parent
		if left < heap.data.len && heap.data[left] < heap.data[smallest] {
			smallest = left
		}
		if right < heap.data.len && heap.data[right] < heap.data[smallest] {
			smallest = right
		}
		if smallest == parent { break }
		heap.data[parent], heap.data[smallest] = heap.data[smallest], heap.data[parent]
		parent = smallest
	}
	return item
}

fn (heap MinHeap[T]) peek() !T {
	if heap.data.len == 0 {
		return error('Heap is empty')
	}
	return heap.data[0]
}

fn (heap MinHeap[T]) len() int {
	return heap.data.len
}

fn (mut heap MinHeap[T]) free() {
	unsafe { heap.data.free() }
}

// Custom max-heap (inverted comparison) that exposes .free().
struct MaxHeap[T] {
mut:
	data []T
}

fn (mut heap MaxHeap[T]) insert(item T) {
	heap.data << item
	mut child := heap.data.len - 1
	mut parent := (child - 1) / 2
	for heap.data[parent] < heap.data[child] {
		heap.data[parent], heap.data[child] = heap.data[child], heap.data[parent]
		child = parent
		parent = (child - 1) / 2
		if child <= 0 { break }
	}
}

fn (mut heap MaxHeap[T]) pop() !T {
	if heap.data.len == 0 {
		return error('Heap is empty')
	} else if heap.data.len == 1 {
		return heap.data.pop()
	}
	item := heap.data[0]
	heap.data[0] = heap.data.pop()
	mut parent := 0
	for {
		left := 2 * parent + 1
		right := 2 * parent + 2
		mut largest := parent
		if left < heap.data.len && heap.data[left] > heap.data[largest] {
			largest = left
		}
		if right < heap.data.len && heap.data[right] > heap.data[largest] {
			largest = right
		}
		if largest == parent { break }
		heap.data[parent], heap.data[largest] = heap.data[largest], heap.data[parent]
		parent = largest
	}
	return item
}

fn (heap MaxHeap[T]) peek() !T {
	if heap.data.len == 0 {
		return error('Heap is empty')
	}
	return heap.data[0]
}

fn (heap MaxHeap[T]) len() int {
	return heap.data.len
}

fn (mut heap MaxHeap[T]) free() {
	unsafe { heap.data.free() }
}

@[manualfree]
fn (mut self HNSW[T]) search_layer(query T, ef int, layer int, entry_points []u64) []u64 {
	mut visited := map[u64]bool{}
	mut candidates := MinHeap[OrdHelper[T]]{}
	mut w := MaxHeap[OrdHelperMax[T]]{}
	mut w_furthest := f64(0)

	for epi in entry_points {
		ep := self.read_value(epi) or { panic(err) }
		d := query.distance_to(ep.value)
		candidates.insert(OrdHelper{ distance: d, value: epi })
		w.insert(OrdHelperMax{ distance: d, value: epi })
		if d > w_furthest { w_furthest = d }
		visited[epi] = true
	}
	for candidates.len() > 0 {
		c := candidates.pop() or { break }
		if c.distance > w_furthest { break
		 }
		cnode := self.read_value(c.value) or { continue }
		for nbi in cnode.neighbours {
			neighbour := self.read_value(nbi) or { panic(err) }
			if neighbour.layer != layer { continue
			 }
			if visited[neighbour.token] { continue
			 }
			visited[neighbour.token] = true

			d := query.distance_to(neighbour.value)
			if d < w_furthest || w.len() < ef {
				candidates.insert(OrdHelper{ distance: d, value: neighbour.token })
				w.insert(OrdHelperMax{ distance: d, value: neighbour.token })
				if w.len() > ef {
					w.pop() or {}
					top := w.peek() or {
						OrdHelperMax[T]{
							distance: 0
							value:    neighbour.token
						}
					}
					w_furthest = top.distance
				} else if d > w_furthest {
					w_furthest = d
				}
			}
		}
	}

	mut ret := []u64{}
	for w.len() > 0 {
		ret << (w.pop() or { break }).value
	}
	unsafe {
		visited.free()
		candidates.free()
		w.free()
	}
	return ret
}

@[manualfree]
fn (mut self HNSW[T]) select_neighbours(node Node[T], candidates []u64, m int) []u64 {
	if candidates.len <= m {
		return candidates
	}
	struct Aux {
		token u64
		dist  f64
	}

	mut scored := []Aux{}
	for c in candidates {
		cn := self.read_value(c) or { panic(err) }
		scored << Aux{
			token: c
			dist:  node.value.distance_to(cn.value)
		}
	}
	scored.sort(a.dist < b.dist)
	ret := scored[..m].map(it.token)
	unsafe {
		scored.free()
	}
	return ret
}

// ------------------------------------------------------------
// Insert  (fixed)
// ------------------------------------------------------------
@[manualfree]
pub fn (mut self HNSW[T]) insert(value T) {
	new_layer := u16(math.floor(-math.log(rand.f64()) * self.normalization_factor))
	mut new_node := self.alloc_node(value, new_layer)
	self.write_node(new_node)

	if !self.has_entry_point {
		self.has_entry_point = true
		self.entry_point = new_node.token
		self.top_layer = new_layer
		return
	}

	mut ep := [self.entry_point]
	l := self.top_layer
	for lc := l; lc > new_layer; lc-- {
		w := self.search_layer(value, 1, lc, ep)
		unsafe { ep.free() }
		ep = [self.nearest(value, w)]
	}

	for lc := math.min(l, new_layer); lc >= 0; lc-- {
		w := self.search_layer(value, self.ef_construction, lc, ep)
		m_cap := if lc == 0 { self.max_neighbours0 } else { self.max_neighbours }
		mut nbrs := self.select_neighbours(new_node, w, self.max_neighbours)

		for mut nbi in nbrs {
			mut nb := self.read_value(nbi) or { panic(err) }
			nb.neighbours << new_node.token
			nb.layer_neighbour_counts[new_node.layer]++
			new_node.neighbours << nb.token
			new_node.layer_neighbour_counts[nb.layer]++
			if nb.neighbours.len > m_cap {
				nb.neighbours = self.select_neighbours(nb, nb.neighbours, m_cap)
			}
			self.write_node(nb)
		}

		unsafe { ep.free() }
		ep = unsafe { w }
	}

	if new_layer > l {
		self.entry_point = new_node.token
		self.top_layer = new_layer
	}
	self.write_node(new_node)
	unsafe { ep.free() }
}

// ------------------------------------------------------------
// KNN search
// ------------------------------------------------------------

@[manualfree]
pub fn (mut self HNSW[T]) knn_search(query T, k int, ef int) []T {
	if !self.has_entry_point {
		return []
	}

	mut ep := [self.entry_point]

	// Greedy descent to layer 1
	for lc := int(self.top_layer); lc > 0; lc-- {
		w := self.search_layer(query, 1, u16(lc), ep)
		unsafe { ep.free() }
		ep = [self.nearest(query, w)]
	}

	// Beam search at layer 0
	candidates := self.search_layer(query, ef, 0, ep)
	struct Aux {
		token u64
		dist  f64
	}

	// Sort by distance and return top-k values
	mut scored := []Aux{}
	for c in candidates {
		node := self.read_value(c) or { continue }
		scored << Aux{
			token: c
			dist:  query.distance_to(node.value)
		}
	}
	scored.sort(a.dist < b.dist)

	take := if scored.len < k { scored.len } else { k }
	mut result := []T{}
	for i in 0 .. take {
		node := self.read_value(scored[i].token) or { continue }
		result << node.value
	}
	unsafe {
		ep.free()
		candidates.free()
		scored.free()
	}
	return result
}

// ------------------------------------------------------------
// Persistence
// ------------------------------------------------------------

pub fn (mut self HNSW[T]) snapshot() ! {
	mut meta := os.open_file('db.meta', 'w') or { return err }
	meta.write_string('${self.has_entry_point}\n') or { return err }
	meta.write_string('${self.entry_point}\n') or { return err }
	meta.write_string('${self.top_layer}\n') or { return err }
	meta.write_string('${self.node_count}\n') or { return err }
	meta.write_string('${self.max_neighbours}\n') or { return err }
	meta.write_string('${self.ef_construction}\n') or { return err }
	meta.close()
}

@[manualfree]
pub fn HNSW.load_from_memory[T]() !HNSW[T] {
	lines := os.read_lines('db.meta')!
	has_ep := lines[0] == 'true'
	ep := lines[1].u64()
	top_layer := u16(lines[2].u16())
	node_count := lines[3].u64()
	max_nb := lines[4].int()
	ef := lines[5].int()

	file := os.open_file('db.nodes', 'r+b')!
	edges := os.open_file('db.edges', 'r+b')!
	ret := HNSW[T]{
		file:                 file
		edges:                edges
		has_entry_point:      has_ep
		entry_point:          ep
		top_layer:            top_layer
		node_count:           node_count
		max_neighbours:       max_nb
		max_neighbours0:      2 * max_nb
		ef_construction:      ef
		normalization_factor: 1.0 / math.log(f64(max_nb))
	}
	unsafe { lines.free() }
	return ret
}
