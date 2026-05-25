module hnsw

import math
import rand
import datatypes

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

struct HNSW[T] {
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

fn new_hnsw[T](capacity int, max_neighbours int, ef_construction int) HNSW[T] {
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
		heap.insert(OrdHelper[T]{ distance: query.distance_to(n.value), value: unsafe { n } })
	}
	mut ret := []T{}
	for ret.len < k {
		ret << (heap.pop() or { break }).value.value
	}
	return ret
}
