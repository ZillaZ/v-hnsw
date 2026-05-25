module hnsw

import math
import arrays
import rand
import datatypes
import os

struct Node[T] {
	token u64
	layer int
mut:
	value      T
	neighbours []&Node[T]
}

struct OrdHelper[T] {
	distance f64
	value    &Node[T]
}

fn (a OrdHelper[T]) < (b OrdHelper[T]) bool {
	return a.distance < b.distance
}

struct OrdHelperMax[T] {
	distance f64
	value    &Node[T]
}

fn (a OrdHelperMax[T]) < (b OrdHelperMax[T]) bool {
	return a.distance > b.distance
}

pub struct HNSW[T] {
mut:
	token                u64
	entry_point          &Node[T] = unsafe { nil }
	top_layer            int      = -1
	ef_construction      int
	normalization_factor f64
	max_neighbours       int
	max_neighbours0      int
	pool                 []Node[T]
}

pub fn new_hnsw[T](capacity int, max_neighbours int, ef_construction int) HNSW[T] {
	return HNSW[T]{
		normalization_factor: 1.0 / math.log(f64(max_neighbours))
		max_neighbours:       max_neighbours
		max_neighbours0:      max_neighbours * 2
		ef_construction:      ef_construction
		pool:                 []Node[T]{cap: capacity}
	}
}

fn (mut self HNSW[T]) alloc_node(value T, layer int) &Node[T] {
	self.token += 1
	self.pool << Node[T]{
		token: self.token
		layer: layer
		value: value
	}
	return &self.pool[self.pool.len - 1]
}

fn (self HNSW[T]) search_layer(query T, ef int, layer int, entry_points []&Node[T]) []&Node[T] {
	mut visited := map[u64]bool{}
	mut candidates := datatypes.MinHeap[OrdHelper[T]]{}
	mut w := datatypes.MinHeap[OrdHelperMax[T]]{}
	mut w_furthest := f64(0)

	for ep in entry_points {
		d := query.distance_to(ep.value)
		candidates.insert(OrdHelper[T]{ distance: d, value: unsafe { ep } })
		w.insert(OrdHelperMax[T]{ distance: d, value: unsafe { ep } })
		if d > w_furthest { w_furthest = d }
		visited[ep.token] = true
	}

	for candidates.len() > 0 {
		c := candidates.pop() or { break }
		if c.distance > w_furthest { break
		 }

		for neighbour in c.value.neighbours {
			if neighbour.layer != layer { continue
			 }
			if visited[neighbour.token] { continue
			 }
			visited[neighbour.token] = true

			d := query.distance_to(neighbour.value)
			if d < w_furthest || w.len() < ef {
				candidates.insert(OrdHelper[T]{ distance: d, value: unsafe { neighbour } })
				w.insert(OrdHelperMax[T]{ distance: d, value: unsafe { neighbour } })
				if w.len() > ef {
					w.pop() or {}
					top := w.peek() or {
						OrdHelperMax[T]{
							distance: 0
							value:    unsafe { neighbour }
						}
					}
					w_furthest = top.distance
				} else if d > w_furthest {
					w_furthest = d
				}
			}
		}
	}

	mut ret := []&Node[T]{}
	for w.len() > 0 {
		ret << (w.pop() or { break }).value
	}
	return ret
}

fn (self HNSW[T]) select_neighbours(base &Node[T], candidates []&Node[T], m int) []&Node[T] {
	mut heap := datatypes.MinHeap[OrdHelper[T]]{}
	for c in candidates {
		heap.insert(OrdHelper[T]{ distance: base.value.distance_to(c.value), value: unsafe { c } })
	}
	mut ret := []&Node[T]{}
	for ret.len < m {
		ret << (heap.pop() or { break }).value
	}
	return ret
}

fn nearest[T](query T, nodes []&Node[T]) &Node[T] {
	mut best_i := 0
	mut best_d := query.distance_to(nodes[0].value)
	for i in 1 .. nodes.len {
		d := query.distance_to(nodes[i].value)
		if d < best_d {
			best_d = d
			best_i = i
		}
	}
	return nodes[best_i]
}

pub fn (mut self HNSW[T]) insert(value T) {
	new_layer := int(math.floor(-math.log(rand.f64()) * self.normalization_factor))
	mut new_node := self.alloc_node(value, new_layer)

	if unsafe { self.entry_point == nil } {
		self.entry_point = new_node
		self.top_layer = new_layer
		return
	}

	mut ep := [self.entry_point]
	l := self.top_layer

	for lc := l; lc > new_layer; lc-- {
		w := self.search_layer(value, 1, lc, ep)
		ep = [nearest(value, w)]
	}

	for lc := math.min(l, new_layer); lc >= 0; lc-- {
		w := self.search_layer(value, self.ef_construction, lc, ep)
		m_cap := if lc == 0 { self.max_neighbours0 } else { self.max_neighbours }
		mut nbrs := self.select_neighbours(new_node, w, self.max_neighbours)

		for mut nb in nbrs {
			nb.neighbours << new_node
			new_node.neighbours << nb
			if nb.neighbours.len > m_cap {
				nb.neighbours = self.select_neighbours(nb, nb.neighbours, m_cap)
			}
		}

		ep = unsafe { w }
	}

	if new_layer > l {
		self.entry_point = new_node
		self.top_layer = new_layer
	}
}

pub fn (self HNSW[T]) knn_search(query T, k int, ef int) []T {
	if unsafe { self.entry_point == nil } { return []T{} }

	mut ep := [self.entry_point]
	for lc := self.top_layer; lc >= 1; lc-- {
		w := self.search_layer(query, 1, lc, ep)
		ep = [nearest(query, w)]
	}

	w := self.search_layer(query, ef, 0, ep)

	mut heap := datatypes.MinHeap[OrdHelper[T]]{}
	for n in w {
		heap.insert(OrdHelper[T]{ distance: query.distance_to(n.value), value: unsafe { n } })
	}
	mut ret := []T{}
	for ret.len < k {
		ret << (heap.pop() or { break }).value.value
	}
	return ret
}

// File layout
// [magic: 8 bytes "HNSW\x00\x00\x00\x01"]
// [token: u64] [top_layer: i64] [ef_construction: i64]
// [normalization_factor: f64] [max_neighbours: i64] [max_neighbours0: i64]
// [pool_len: u64]
// for each node:
//   [token: u64] [layer: i64] [value: T serialized via T.to_bytes()]
// [entry_point_token: u64]  (0 = nil)
// for each node (same order):
//   [neighbour_count: u64] [neighbour_token_0: u64] ...

const snapshot_magic = arrays.merge('HNSW'.bytes(), [u8(0), u8(0), u8(0), u8(1)])

// T must implement:
//   fn (v T) to_bytes() []u8
//   fn T.from_bytes(b []u8) T
//   fn T.byte_size() int   // fixed size in bytes

pub fn (self HNSW[T]) snapshot(path string) ! {
	mut f := os.create(path)!
	defer { f.close() }

	// magic
	f.write(snapshot_magic)!

	// header scalars
	write_u64(mut f, self.token)!
	write_i64(mut f, i64(self.top_layer))!
	write_i64(mut f, i64(self.ef_construction))!
	write_f64(mut f, self.normalization_factor)!
	write_i64(mut f, i64(self.max_neighbours))!
	write_i64(mut f, i64(self.max_neighbours0))!

	// pool
	write_u64(mut f, u64(self.pool.len))!
	for node in self.pool {
		write_u64(mut f, node.token)!
		write_i64(mut f, i64(node.layer))!
		f.write(node.value.to_bytes())!
	}

	// entry point token (0 means nil)
	ep_token := if unsafe { self.entry_point == nil } { u64(0) } else { self.entry_point.token }
	write_u64(mut f, ep_token)!

	// adjacency lists — store only tokens; we'll resolve to pointers on load
	for node in self.pool {
		write_u64(mut f, u64(node.neighbours.len))!
		for nb in node.neighbours {
			write_u64(mut f, nb.token)!
		}
	}
}

pub fn load_hnsw_snapshot[T](path string) !HNSW[T] {
	mut f := os.open(path)!
	defer { f.close() }

	// verify magic
	mut magic := []u8{len: 8}
	f.read(mut magic)!
	for i, b in snapshot_magic {
		if magic[i] != b {
			return error('hnsw snapshot: bad magic bytes — file may be corrupt or wrong format')
		}
	}

	// header
	token := read_u64(mut f)!
	top_layer := int(read_i64(mut f)!)
	ef_construction := int(read_i64(mut f)!)
	norm_factor := read_f64(mut f)!
	max_neighbours := int(read_i64(mut f)!)
	max_neighbours0 := int(read_i64(mut f)!)

	pool_len := read_u64(mut f)!
	val_size := T.byte_size()

	mut hnsw := HNSW[T]{
		token:                token
		top_layer:            top_layer
		ef_construction:      ef_construction
		normalization_factor: norm_factor
		max_neighbours:       max_neighbours
		max_neighbours0:      max_neighbours0
		pool:                 []Node[T]{cap: int(pool_len)}
	}

	// read nodes (values only; neighbours come later)
	for _ in 0 .. pool_len {
		tok := read_u64(mut f)!
		layer := int(read_i64(mut f)!)
		mut vbuf := []u8{len: val_size}
		f.read(mut vbuf)!
		hnsw.pool << Node[T]{
			token: tok
			layer: layer
			value: T.from_bytes(vbuf)
		}
	}

	// entry point
	ep_token := read_u64(mut f)!
	if ep_token != 0 {
		for i in 0 .. hnsw.pool.len {
			if hnsw.pool[i].token == ep_token {
				hnsw.entry_point = &hnsw.pool[i]
				break
			}
		}
	}

	// build token → pool-index map for O(1) neighbour resolution
	mut tok_idx := map[u64]int{}
	for i, node in hnsw.pool {
		tok_idx[node.token] = i
	}

	// adjacency lists
	for i in 0 .. hnsw.pool.len {
		nb_count := read_u64(mut f)!
		for _ in 0 .. nb_count {
			nb_tok := read_u64(mut f)!
			idx := tok_idx[nb_tok] or {
				return error('hnsw snapshot: unknown neighbour token ${nb_tok}')
			}
			hnsw.pool[i].neighbours << &hnsw.pool[idx]
		}
	}

	return hnsw
}

// ── tiny helpers ─────────────────────────────────────────────────────────────

fn write_u64(mut f os.File, v u64) ! {
	mut b := [u8(0), 0, 0, 0, 0, 0, 0, 0]
	b[0] = u8(v >> 56)
	b[1] = u8(v >> 48)
	b[2] = u8(v >> 40)
	b[3] = u8(v >> 32)
	b[4] = u8(v >> 24)
	b[5] = u8(v >> 16)
	b[6] = u8(v >> 8)
	b[7] = u8(v)
	f.write(b)!
}

fn write_i64(mut f os.File, v i64) ! {
	write_u64(mut f, u64(v))!
}

fn write_f64(mut f os.File, v f64) ! {
	bits := unsafe { *(&u64(&v)) } // reinterpret bits
	write_u64(mut f, bits)!
}

fn read_u64(mut f os.File) !u64 {
	mut b := [u8(0), 0, 0, 0, 0, 0, 0, 0]
	f.read(mut b)!
	return (u64(b[0]) << 56) | (u64(b[1]) << 48) | (u64(b[2]) << 40) | (u64(b[3]) << 32) | (u64(b[4]) << 24) | (u64(b[5]) << 16) | (u64(b[6]) << 8) | u64(b[7])
}

fn read_i64(mut f os.File) !i64 {
	return i64(read_u64(mut f)!)
}

fn read_f64(mut f os.File) !f64 {
	bits := read_u64(mut f)!
	return unsafe { *(&f64(&bits)) }
}
