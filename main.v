module main

import math
import rand
import datatypes

@[heap]
struct Node[T] {
	token u64
	layer int
mut:
	value      T
	neighbours []&Node[T]
}

@[heap]
struct OrdHelper[T] {
	distance f64
	value    &Node[T]
}

fn (a OrdHelper[T]) < (b OrdHelper[T]) bool {
	return a.distance < b.distance
}

@[heap]
struct OrdHelperMax[T] {
	distance f64
	value    &Node[T]
}

fn (a OrdHelperMax[T]) < (b OrdHelperMax[T]) bool {
	return a.distance > b.distance
}

// HNSW owns all nodes via `pool`. The pool is pre-allocated to `capacity`
// so it never reallocates — meaning &pool[i] pointers are stable forever.
struct HNSW[T] {
mut:
	token                u64
	entry_point          &Node[T] = unsafe { nil }
	top_layer            int      = -1
	ef_construction      int
	normalization_factor f64
	max_neighbours       int
	max_neighbours0      int
	pool                 []Node[T] // GC root — owns every node
}

fn new_hnsw[T](capacity int, max_neighbours int, ef_construction int) HNSW[T] {
	return HNSW[T]{
		normalization_factor: 1.0 / math.log(f64(max_neighbours))
		max_neighbours:       max_neighbours
		max_neighbours0:      max_neighbours * 2
		ef_construction:      ef_construction
		pool:                 []Node[T]{cap: capacity}
	}
}

// alloc_node appends to the pool and returns a stable pointer into it.
// Safe as long as pool.len never exceeds the cap set in new_hnsw.
fn (mut self HNSW[T]) alloc_node(value T, layer int) &Node[T] {
	self.token += 1
	self.pool << Node[T]{
		token: self.token
		layer: layer
		value: value
	}
	return &self.pool[self.pool.len - 1]
}

// Algorithm 2 — SEARCH-LAYER
fn (self HNSW[T]) search_layer(query T, ef int, layer int, entry_points []&Node[T]) []&Node[T] {
	mut visited := map[u64]bool{}
	mut candidates := datatypes.MinHeap[OrdHelper[T]]{}
	mut w := datatypes.MinHeap[OrdHelperMax[T]]{}
	mut w_furthest := f64(0)

	for ep in entry_points {
		d := query.distance_to(ep.value)
		candidates.insert(OrdHelper[T]{ distance: d, value: ep })
		w.insert(OrdHelperMax[T]{ distance: d, value: ep })
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
				candidates.insert(OrdHelper[T]{ distance: d, value: neighbour })
				w.insert(OrdHelperMax[T]{ distance: d, value: neighbour })
				if w.len() > ef {
					w.pop() or {}
					top := w.peek() or {
						OrdHelperMax[T]{
							distance: 0
							value:    neighbour
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

// Algorithm 3 — SELECT-NEIGHBORS-SIMPLE
fn (self HNSW[T]) select_neighbours(base &Node[T], candidates []&Node[T], m int) []&Node[T] {
	mut heap := datatypes.MinHeap[OrdHelper[T]]{}
	for c in candidates {
		heap.insert(OrdHelper[T]{ distance: base.value.distance_to(c.value), value: c })
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

// Algorithm 1 — INSERT
fn (mut self HNSW[T]) insert(value T) {
	new_layer := int(math.floor(-math.log(rand.f64()) * self.normalization_factor))
	mut new_node := self.alloc_node(value, new_layer)

	if unsafe { self.entry_point == nil } {
		self.entry_point = new_node
		self.top_layer = new_layer
		return
	}

	mut ep := [self.entry_point]
	l := self.top_layer

	// Phase 1: above new_layer, greedy descent ef=1
	for lc := l; lc > new_layer; lc-- {
		w := self.search_layer(value, 1, lc, ep)
		ep = [nearest(value, w)]
	}

	// Phase 2: 0..new_layer, full search + connect
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

// Algorithm 5 — K-NN-SEARCH
fn (self HNSW[T]) knn_search(query T, k int, ef int) []T {
	if unsafe { self.entry_point == nil } { return []T{} }

	mut ep := [self.entry_point]
	for lc := self.top_layer; lc >= 1; lc-- {
		w := self.search_layer(query, 1, lc, ep)
		ep = [nearest(query, w)]
	}

	w := self.search_layer(query, ef, 0, ep)

	mut heap := datatypes.MinHeap[OrdHelper[T]]{}
	for n in w {
		heap.insert(OrdHelper[T]{ distance: query.distance_to(n.value), value: n })
	}
	mut ret := []T{}
	for ret.len < k {
		ret << (heap.pop() or { break }).value.value
	}
	return ret
}

// ---- Example value type ----

struct IntWrapper {
	value int
}

fn (a IntWrapper) distance_to(b IntWrapper) f64 {
	return math.abs(f64(a.value) - f64(b.value))
}

fn (a IntWrapper) < (b IntWrapper) bool {
	return a.value < b.value
}

fn main() {
	mut hnsw := new_hnsw[IntWrapper](1000, 16, 200)
	for i in 0 .. 1000 {
		hnsw.insert(IntWrapper{ value: i })
	}
	println('5 nearest neighbours of 10 (ef=50):')
	println(hnsw.knn_search(IntWrapper{ value: 10 }, 5, 50))
}
