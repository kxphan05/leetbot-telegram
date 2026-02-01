import logging
from typing import Optional, List
from sqlalchemy import select, func
from models import Algorithm
from database.database import AsyncSessionLocal
from database.db_models import AlgorithmDB

logger = logging.getLogger(__name__)


class AlgorithmService:
    def __init__(self):
        self._use_database = True

    def _db_to_model(self, db_algo: AlgorithmDB) -> Algorithm:
        """Convert database model to Pydantic model."""
        return Algorithm(
            id=db_algo.id,
            name=db_algo.name,
            category=db_algo.category,
            description=db_algo.description,
            implementation=db_algo.implementation,
            complexity=db_algo.complexity,
            use_cases=db_algo.use_cases or [],
            external_link=db_algo.external_link,
        )

    async def get_all_algorithms(self) -> List[Algorithm]:
        """Get all algorithms from the database."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AlgorithmDB))
                return [self._db_to_model(a) for a in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Database query failed, using fallback: {e}")
            return self._get_fallback_algorithms()

    async def search_algorithms(self, query: str) -> List[Algorithm]:
        """Search algorithms by name, category, or description."""
        try:
            async with AsyncSessionLocal() as session:
                search_pattern = f"%{query.lower()}%"
                result = await session.execute(
                    select(AlgorithmDB)
                    .where(
                        (func.lower(AlgorithmDB.name).like(search_pattern))
                        | (func.lower(AlgorithmDB.category).like(search_pattern))
                        | (func.lower(AlgorithmDB.description).like(search_pattern))
                    )
                    .limit(10)
                )
                return [self._db_to_model(a) for a in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Failed to search algorithms: {e}")
            fallback = self._get_fallback_algorithms()
            query_lower = query.lower()
            return [
                a
                for a in fallback
                if query_lower in a.name.lower()
                or query_lower in a.category.lower()
                or query_lower in a.description.lower()
            ]

    async def get_algorithm_by_category(self, category: str) -> List[Algorithm]:
        """Get algorithms filtered by category."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AlgorithmDB).where(
                        func.lower(AlgorithmDB.category) == category.lower()
                    )
                )
                return [self._db_to_model(a) for a in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Failed to get algorithms by category: {e}")
            fallback = self._get_fallback_algorithms()
            return [a for a in fallback if a.category.lower() == category.lower()]

    async def get_algorithm(self, algorithm_id: int) -> Optional[Algorithm]:
        """Get a single algorithm by ID."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AlgorithmDB).where(AlgorithmDB.id == algorithm_id)
                )
                db_algo = result.scalar_one_or_none()
                if db_algo:
                    return self._db_to_model(db_algo)
        except Exception as e:
            logger.warning(f"Failed to get algorithm: {e}")
        fallback = self._get_fallback_algorithms()
        for algo in fallback:
            if algo.id == algorithm_id:
                return algo
        return None

    async def get_categories(self) -> List[str]:
        """Get all unique algorithm categories."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AlgorithmDB.category).distinct())
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get categories: {e}")
            fallback = self._get_fallback_algorithms()
            return list(set(a.category for a in fallback))

    def _get_fallback_algorithms(self) -> List[Algorithm]:
        """Return fallback algorithms when database is unavailable."""
        return [
            Algorithm(
                id=1,
                name="Binary Search",
                category="Search",
                description="A search algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search interval in half.",
                implementation="""def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
                complexity="O(log n)",
                use_cases=[
                    "Finding elements in sorted arrays",
                    "Finding insertion points",
                ],
                external_link="https://en.wikipedia.org/wiki/Binary_search_algorithm",
            ),
            Algorithm(
                id=2,
                name="Dijkstra's Algorithm",
                category="Graph",
                description="A graph traversal algorithm that finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights.",
                implementation="""import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        curr_dist, node = heapq.heappop(pq)
        if curr_dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances""",
                complexity="O((V + E) log V)",
                use_cases=[
                    "Finding shortest paths in road networks",
                    "Network routing protocols",
                    "GPS navigation systems",
                ],
                external_link="https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm",
            ),
            Algorithm(
                id=3,
                name="Breadth-First Search",
                category="Graph",
                description="A graph traversal algorithm that explores vertices in order of their distance from the source vertex, visiting all neighbors before moving to the next level.",
                implementation="""from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)""",
                complexity="O(V + E)",
                use_cases=[
                    "Finding shortest path in unweighted graphs",
                    "Level-order tree traversal",
                    "Web crawlers",
                ],
                external_link="https://en.wikipedia.org/wiki/Breadth-first_search",
            ),
            Algorithm(
                id=4,
                name="Depth-First Search",
                category="Graph",
                description="A graph traversal algorithm that explores as far as possible along each branch before backtracking.",
                implementation="""def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited""",
                complexity="O(V + E)",
                use_cases=[
                    "Topological sorting",
                    "Finding connected components",
                    "Solving mazes and puzzles",
                ],
                external_link="https://en.wikipedia.org/wiki/Depth-first_search",
            ),
            Algorithm(
                id=5,
                name="Quick Sort",
                category="Sorting",
                description="A divide-and-conquer sorting algorithm that picks a pivot element and partitions the array around the pivot.",
                implementation="""def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",
                complexity="O(n log n) average, O(n^2) worst",
                use_cases=[
                    "General purpose sorting",
                    "In-memory sorting",
                    "External sorting with limited memory",
                ],
                external_link="https://en.wikipedia.org/wiki/Quicksort",
            ),
            Algorithm(
                id=6,
                name="Merge Sort",
                category="Sorting",
                description="A divide-and-conquer sorting algorithm that divides the array into halves, sorts them, and then merges them.",
                implementation="""def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result""",
                complexity="O(n log n)",
                use_cases=[
                    "External sorting (large datasets)",
                    "Stable sorting required",
                    "Linked list sorting",
                ],
                external_link="https://en.wikipedia.org/wiki/Merge_sort",
            ),
            Algorithm(
                id=7,
                name="Bubble Sort",
                category="Sorting",
                description="A simple sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order.",
                implementation="""def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
                complexity="O(n^2)",
                use_cases=[
                    "Educational purposes",
                    "Small nearly-sorted arrays",
                ],
                external_link="https://en.wikipedia.org/wiki/Bubble_sort",
            ),
            Algorithm(
                id=8,
                name="Linear Search",
                category="Search",
                description="A sequential search algorithm that checks each element of the list until the target is found or the list ends.",
                implementation="""def linear_search(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1""",
                complexity="O(n)",
                use_cases=[
                    "Unsorted arrays",
                    "Small datasets",
                    "One-time searches",
                ],
                external_link="https://en.wikipedia.org/wiki/Linear_search",
            ),
            Algorithm(
                id=9,
                name="Kadane's Algorithm",
                category="Dynamic Programming",
                description="An algorithm for finding the maximum subarray sum in a one-dimensional array of numbers.",
                implementation="""def max_subarray(arr):
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum""",
                complexity="O(n)",
                use_cases=[
                    "Maximum subarray problems",
                    "Stock price analysis",
                    "Pattern recognition",
                ],
                external_link="https://en.wikipedia.org/wiki/Maximum_subarray_problem",
            ),
            Algorithm(
                id=10,
                name="Floyd's Cycle Detection",
                category="Linked List",
                description="An algorithm that uses two pointers moving at different speeds to detect cycles in a linked list or determine the length of a cycle.",
                implementation="""def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def cycle_length(head):
    slow = fast = head
    count = 0
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = slow.next
            count = 1
            while slow != fast:
                slow = slow.next
                count += 1
            return count
    return 0""",
                complexity="O(n)",
                use_cases=[
                    "Detecting cycles in linked lists",
                    "Finding cycle length",
                    "Finding cycle start node",
                ],
                external_link="https://en.wikipedia.org/wiki/Cycle_detection",
            ),
            Algorithm(
                id=11,
                name="Hash Table",
                category="Data Structure",
                description="A data structure that implements an associative array, mapping keys to values using a hash function.",
                implementation="""class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def hash_function(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        index = self.hash_function(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
    
    def search(self, key):
        index = self.hash_function(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None""",
                complexity="O(1) average, O(n) worst",
                use_cases=[
                    "Database indexing",
                    "Caching",
                    "Symbol tables",
                ],
                external_link="https://en.wikipedia.org/wiki/Hash_table",
            ),
            Algorithm(
                id=12,
                name="Heap Sort",
                category="Sorting",
                description="A comparison-based sorting algorithm that uses a binary heap data structure to sort elements.",
                implementation="""import heapq

def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]""",
                complexity="O(n log n)",
                use_cases=[
                    "Priority queue implementation",
                    "Top-k element selection",
                    "Partial sorting",
                ],
                external_link="https://en.wikipedia.org/wiki/Heapsort",
            ),
            Algorithm(
                id=13,
                name="A* Search",
                category="Graph",
                description="A graph traversal and pathfinding algorithm that uses heuristics to find the shortest path efficiently.",
                implementation="""import heapq

def a_star(graph, start, goal, heuristic):
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor in graph[current]:
            tentative_g = g_score[current] + graph[current][neighbor]
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None""",
                complexity="O(E) with optimal heuristic",
                use_cases=[
                    "Pathfinding in games",
                    "GPS navigation",
                    "Robot motion planning",
                ],
                external_link="https://en.wikipedia.org/wiki/A*_search_algorithm",
            ),
            Algorithm(
                id=14,
                name="Bellman-Ford Algorithm",
                category="Graph",
                description="An algorithm that finds shortest paths from a single source vertex to all other vertices in a weighted graph, handling negative edge weights.",
                implementation="""def bellman_ford(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] != float('inf'):
                    if distances[node] + weight < distances[neighbor]:
                        distances[neighbor] = distances[node] + weight
    return distances""",
                complexity="O(V * E)",
                use_cases=[
                    "Negative weight edges",
                    "Currency arbitrage detection",
                    "Network routing with negative costs",
                ],
                external_link="https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm",
            ),
            Algorithm(
                id=15,
                name="Topological Sort",
                category="Graph",
                description="A linear ordering of vertices in a directed acyclic graph (DAG) where for every directed edge uv, vertex u comes before v.",
                implementation="""from collections import deque, defaultdict

def topological_sort(graph):
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([node for node in graph if in_degree[node] == 0])
    topo_order = []
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return topo_order if len(topo_order) == len(graph) else None""",
                complexity="O(V + E)",
                use_cases=[
                    "Task scheduling",
                    "Build system dependency resolution",
                    "Course scheduling",
                ],
                external_link="https://en.wikipedia.org/wiki/Topological_sorting",
            ),
            Algorithm(
                id=16,
                name="Rabin-Karp Algorithm",
                category="String",
                description="A string-searching algorithm that uses hashing to find any of a set of pattern strings in a text.",
                implementation="""def rabin_karp(text, pattern):
    if not pattern or not text:
        return -1
    base = 26
    mod = 10**9 + 7
    m, n = len(pattern), len(text)
    
    pattern_hash = sum(ord(pattern[i]) * pow(base, m - 1 - i, mod) for i in range(m)) % mod
    text_hash = sum(ord(text[i]) * pow(base, m - 1 - i, mod) for i in range(m)) % mod
    
    for i in range(n - m + 1):
        if pattern_hash == text_hash and text[i:i+m] == pattern:
            return i
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * pow(base, m - 1, mod)) + ord(text[i + m])) % mod
    return -1""",
                complexity="O(n + m) average, O(n * m) worst",
                use_cases=[
                    "Plagiarism detection",
                    "String searching in large texts",
                    "DNA sequence matching",
                ],
                external_link="https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm",
            ),
            Algorithm(
                id=17,
                name="KMP Algorithm",
                category="String",
                description="The Knuth-Morris-Pratt algorithm for string searching that uses a precomputed prefix function to skip unnecessary comparisons.",
                implementation="""def compute_lps(pattern):
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps

def kmp_search(text, pattern):
    if not pattern or not text:
        return -1
    lps = compute_lps(pattern)
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        elif j != 0:
            j = lps[j - 1]
        else:
            i += 1
        if j == len(pattern):
            return i - j
    return -1""",
                complexity="O(n + m)",
                use_cases=[
                    "Efficient string searching",
                    "Pattern matching in DNA sequences",
                    "Text editor find function",
                ],
                external_link="https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm",
            ),
            Algorithm(
                id=18,
                name="Union-Find",
                category="Graph",
                description="A data structure that keeps track of a partition of elements into disjoint sets and supports union and find operations.",
                implementation="""class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True""",
                complexity="O(alpha(n)) amortized",
                use_cases=[
                    "Kruskal's MST algorithm",
                    "Network connectivity",
                    "Image processing (connected components)",
                ],
                external_link="https://en.wikipedia.org/wiki/Disjoint-set_data_structure",
            ),
            Algorithm(
                id=19,
                name="Sliding Window",
                category="Two Pointers",
                description="A technique for efficiently solving problems that involve arrays or strings by maintaining a window that slides through the data.",
                implementation="""def longest_substring_k_distinct(s, k):
    if k == 0:
        return 0
    from collections import defaultdict
    char_count = defaultdict(int)
    left = 0
    max_len = 0
    distinct = 0
    
    for right in range(len(s)):
        char_count[s[right]] += 1
        if char_count[s[right]] == 1:
            distinct += 1
        while distinct > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                distinct -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len""",
                complexity="O(n)",
                use_cases=[
                    "Maximum subarray problems",
                    "Substring with conditions",
                    "Sliding window minimum/maximum",
                ],
                external_link="https://en.wikipedia.org/wiki/Sliding_window",
            ),
            Algorithm(
                id=20,
                name="Trie",
                category="Data Structure",
                description="A tree-like data structure used to store strings efficiently, where each node represents a character.",
                implementation="""class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True""",
                complexity="O(m) for operations (m = key length)",
                use_cases=[
                    "Autocomplete",
                    "Spell checking",
                    "IP routing (longest prefix match)",
                ],
                external_link="https://en.wikipedia.org/wiki/Trie",
            ),
        ]


algorithm_service = AlgorithmService()
