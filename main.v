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

struct HNSW[T] {
mut:
	token                u64
	entry_point          &Node[T]
	nodes                map[int][]&Node[T]
	normalization_factor f64
	max_neighbours       int
}

fn (mut self HNSW[T]) insert(value T) {
	self.token += 1
	new_element_layer := int(math.floor(-math.log(rand.f64()) * self.normalization_factor))
	mut new_node := Node{
		token:      self.token
		layer:      new_element_layer
		value:      value
		neighbours: []
	}
	if mut v := self.nodes[new_element_layer] {
		v << &new_node
		self.nodes[new_element_layer] = v
	} else {
		self.nodes[new_element_layer] = [&new_node]
	}
	if unsafe { self.entry_point == 0 } {
		self.entry_point = &new_node
	}
	mut entry_point := self.entry_point
	if self.entry_point.layer < new_element_layer + 1 {
		for layer in self.entry_point.layer .. new_element_layer + 1 {
			nodes := self.search_layer(value, 1, layer, [entry_point])
			mut heap := datatypes.MinHeap[OrdHelper[T]]{}
			for node in nodes {
				heap.insert(OrdHelper{
					value:    node
					distance: value - node.value
				})
			}
			entry_point = heap.pop() or { continue }.value
		}
	}
	mut layer := math.min(self.entry_point.layer, new_element_layer)
	for layer >= 0 {
		nodes := self.search_layer(value, 1, layer, [entry_point])
		mut neighbours := self.select_neighbours_simple(new_node, nodes, self.max_neighbours, layer)
		for mut neighbour in neighbours {
			neighbour.neighbours << &new_node
			new_node.neighbours << neighbour
			if neighbour.neighbours.len > self.max_neighbours {
				neighbour.neighbours = self.select_neighbours_simple(neighbour,
					neighbour.neighbours, self.max_neighbours, neighbour.layer)
			}
		}
		entry_point = nodes[0]
		layer -= 1
	}
	if self.entry_point.layer < new_element_layer {
		self.entry_point = &new_node
	}
}

@[heap]
struct OrdHelper[T] {
	distance f64
	value    &Node[T]
}

fn (a OrdHelper[T]) < (b OrdHelper[T]) bool {
	return a.distance < b.distance
}

fn (self HNSW[T]) select_neighbours_simple(value &Node[T], candidates []&Node[T], count int, layer int) []&Node[T] {
	mut heap := datatypes.MinHeap[OrdHelper[T]]{}
	for candidate in candidates {
		helper := OrdHelper{
			distance: value.value - candidate.value
			value:    candidate
		}
		heap.insert(helper)
	}
	mut ret := []&Node[T]{}
	for _ in 0 .. count {
		ret << heap.pop() or { break }.value
	}
	return ret
}

fn (self HNSW[T]) search_layer(value T, count int, layer int, entry_point []&Node[T]) []&Node[T] {
	mut visited := unsafe { entry_point }
	mut candidates := unsafe { entry_point }
	mut ret := unsafe { entry_point }
	for _ in 0 .. candidates.len {
		mut closest := OrdHelper{
			value:    candidates[0]
			distance: value - candidates[0].value
		}
		mut furthest := closest
		for candidate in candidates {
			aux := OrdHelper{
				value:    candidate
				distance: value - candidate.value
			}
			if aux < closest {
				closest = aux
			}
			if aux > furthest {
				furthest = aux
			}
		}
		if closest.distance > furthest.distance {
			break
		}
		mut nn := []&Node[T]{}
		for neighbour in closest.value.neighbours {
			if neighbour.layer != layer { continue
			 }
			nn << neighbour
		}
		for neighbour in nn {
			mut v := false
			for vi in visited {
				if vi != neighbour { continue
				 }
				v = true
			}
			if !v {
				visited << neighbour
			}
			furthest = OrdHelper{
				value:    ret[0]
				distance: value - ret[0].value
			}
			for r in ret {
				aux := OrdHelper{
					value:    r
					distance: value - r.value
				}
				if aux > furthest {
					furthest = aux
				}
			}
			aux2 := OrdHelper{
				value:    neighbour
				distance: value - neighbour.value
			}
			if aux2 < furthest || ret.len < count {
				candidates << neighbour
				ret << neighbour
			}

			if ret.len > count {
				mut nret := []&Node[T]{}
				for r in ret {
					if r == furthest.value {
						continue
					}
					nret << r
				}
				ret = unsafe { nret }
			}
		}
	}
	return ret
}

fn (self HNSW[T]) knn_search(target T, count int, size int) []T {
	mut ret := []T{}
	mut entry_point := self.entry_point
	mut layer := entry_point.layer
	for layer >= 1 {
		mut candidates := self.search_layer(target, count, layer, [entry_point])
		mut smallest := candidates[0]
		for mut candidate in candidates {
			if target - candidate.value < target - smallest.value {
				smallest = candidate
			}
		}
		entry_point = smallest
		layer -= 1
	}
	aux := self.search_layer(target, count, 0, [entry_point])
	mut heap := datatypes.MinHeap[OrdHelper[T]]{}
	for a in aux {
		heap.insert(OrdHelper{
			value:    a
			distance: target - a.value
		})
	}
	for _ in 0 .. size {
		ret << heap.pop() or { break }.value.value
	}
	return ret
}

fn main() {
	mut hnsw := HNSW[int]{
		normalization_factor: 5.0
		max_neighbours:       5
	}
	for i in 0 .. 1000 {
		hnsw.insert(i)
	}
	println(hnsw.knn_search(10, 50, 5))
}
