"""Seed data for questions and algorithms."""

import asyncio
import logging
from sqlalchemy import select
from database.database import AsyncSessionLocal, init_db
from database.db_models import QuestionDB, AlgorithmDB

logger = logging.getLogger(__name__)

SEED_QUESTIONS = [
    {
        "leetcode_id": 1,
        "title": "Two Sum",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "solution": """def twoSum(nums, target):
    prevMap = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[n] = i
    return""",
        "solution_approach": "Hash Map",
        "examples": [
            {
                "input": "nums = [2,7,11,15], target = 9",
                "output": "[0,1]",
                "explanation": "Because nums[0] + nums[1] == 9, we return [0, 1].",
            },
            {
                "input": "nums = [3,2,4], target = 6",
                "output": "[1,2]",
                "explanation": "Because nums[1] + nums[2] == 6, we return [1, 2].",
            },
        ],
        "external_link": "https://leetcode.com/problems/two-sum/",
    },
    {
        "leetcode_id": 2,
        "title": "Add Two Numbers",
        "difficulty": "Medium",
        "category": "Linked List",
        "description": "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit.",
        "solution": """def addTwoNumbers(l1, l2):
    dummy = ListNode()
    cur = dummy
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        val = v1 + v2 + carry
        carry = val // 10
        val = val % 10
        cur.next = ListNode(val)
        cur = cur.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next""",
        "solution_approach": "Linked List Traversal with Carry",
        "examples": [
            {
                "input": "l1 = [2,4,3], l2 = [5,6,4]",
                "output": "[7,0,8]",
                "explanation": "342 + 465 = 807.",
            }
        ],
        "external_link": "https://leetcode.com/problems/add-two-numbers/",
    },
    {
        "leetcode_id": 3,
        "title": "Longest Substring Without Repeating Characters",
        "difficulty": "Medium",
        "category": "Sliding Window",
        "description": "Given a string s, find the length of the longest substring without repeating characters.",
        "solution": """def lengthOfLongestSubstring(s):
    charSet = set()
    left = 0
    result = 0
    for right in range(len(s)):
        while s[right] in charSet:
            charSet.remove(s[left])
            left += 1
        charSet.add(s[right])
        result = max(result, right - left + 1)
    return result""",
        "solution_approach": "Sliding Window with Hash Set",
        "examples": [
            {
                "input": 's = "abcabcbb"',
                "output": "3",
                "explanation": "The answer is 'abc', with the length of 3.",
            },
            {
                "input": 's = "bbbbb"',
                "output": "1",
                "explanation": "The answer is 'b', with the length of 1.",
            },
        ],
        "external_link": "https://leetcode.com/problems/longest-substring-without-repeating-characters/",
    },
    {
        "leetcode_id": 4,
        "title": "Median of Two Sorted Arrays",
        "difficulty": "Hard",
        "category": "Binary Search",
        "description": "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
        "solution": """def findMedianSortedArrays(nums1, nums2):
    A, B = nums1, nums2
    total = len(A) + len(B)
    half = total // 2
    if len(A) < len(B):
        A, B = B, A
    l, r = 0, len(A) - 1
    while True:
        i = (l + r) // 2
        j = half - i - 2
        Aleft = A[i] if i >= 0 else float("-inf")
        Aright = A[i + 1] if i + 1 < len(A) else float("inf")
        Bleft = B[j] if j >= 0 else float("-inf")
        Bright = B[j + 1] if j + 1 < len(B) else float("inf")
        if Aleft <= Bright and Bleft <= Aright:
            if total % 2:
                return min(Aright, Bright)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1""",
        "solution_approach": "Binary Search",
        "examples": [
            {
                "input": "nums1 = [1,3], nums2 = [2]",
                "output": "2.00000",
                "explanation": "The median is 2.0.",
            },
            {
                "input": "nums1 = [1,2], nums2 = [3,4]",
                "output": "2.50000",
                "explanation": "The median is (2 + 3) / 2 = 2.5.",
            },
        ],
        "external_link": "https://leetcode.com/problems/median-of-two-sorted-arrays/",
    },
    {
        "leetcode_id": 5,
        "title": "Longest Palindromic Substring",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given a string s, return the longest palindromic substring in s.",
        "solution": """def longestPalindrome(s):
    if len(s) < 2:
        return s
    start, max_len = 0, 1
    for i in range(len(s)):
        low, high = i - 1, i + 1
        while high < len(s) and s[i] == s[high]:
            high += 1
        while low >= 0 and s[i] == s[low]:
            low -= 1
        while low >= 0 and high < len(s) and s[low] == s[high]:
            low -= 1
            high += 1
        length = high - low - 1
        if length > max_len:
            start = low + 1
            max_len = length
    return s[start:start + max_len]""",
        "solution_approach": "Expand Around Center",
        "examples": [
            {
                "input": 's = "babad"',
                "output": '"bab" or "aba"',
                "explanation": "Three palindromes: 'bab', 'aba', 'bb'.",
            },
            {
                "input": 's = "cbbd"',
                "output": '"bb"',
                "explanation": "The longest palindrome is 'bb'.",
            },
        ],
        "external_link": "https://leetcode.com/problems/longest-palindromic-substring/",
    },
    {
        "leetcode_id": 121,
        "title": "Best Time to Buy and Sell Stock",
        "difficulty": "Easy",
        "category": "Array",
        "description": "You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.",
        "solution": """def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit""",
        "solution_approach": "One Pass with Min Tracking",
        "examples": [
            {
                "input": "prices = [7,1,5,3,6,4]",
                "output": "5",
                "explanation": "Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.",
            }
        ],
        "external_link": "https://leetcode.com/problems/best-time-to-buy-and-sell-stock/",
    },
    {
        "leetcode_id": 20,
        "title": "Valid Parentheses",
        "difficulty": "Easy",
        "category": "Stack",
        "description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. An input string is valid if open brackets are closed by the same type of brackets and in the correct order.",
        "solution": """def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return len(stack) == 0""",
        "solution_approach": "Stack",
        "examples": [
            {
                "input": 's = "()"',
                "output": "true",
                "explanation": "Open bracket is closed by same type.",
            },
            {
                "input": 's = "()[]{}"',
                "output": "true",
                "explanation": "All brackets properly closed.",
            },
            {
                "input": 's = "(]"',
                "output": "false",
                "explanation": "Mismatched bracket types.",
            },
        ],
        "external_link": "https://leetcode.com/problems/valid-parentheses/",
    },
    {
        "leetcode_id": 21,
        "title": "Merge Two Sorted Lists",
        "difficulty": "Easy",
        "category": "Linked List",
        "description": "You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list by splicing together the nodes of the first two lists.",
        "solution": """def mergeTwoLists(list1, list2):
    dummy = ListNode()
    tail = dummy
    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    tail.next = list1 or list2
    return dummy.next""",
        "solution_approach": "Two Pointers",
        "examples": [
            {
                "input": "list1 = [1,2,4], list2 = [1,3,4]",
                "output": "[1,1,2,3,4,4]",
                "explanation": "Merged in sorted order.",
            }
        ],
        "external_link": "https://leetcode.com/problems/merge-two-sorted-lists/",
    },
    {
        "leetcode_id": 53,
        "title": "Maximum Subarray",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given an integer array nums, find the subarray with the largest sum, and return its sum.",
        "solution": """def maxSubArray(nums):
    max_sum = nums[0]
    current_sum = nums[0]
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum""",
        "solution_approach": "Kadane's Algorithm",
        "examples": [
            {
                "input": "nums = [-2,1,-3,4,-1,2,1,-5,4]",
                "output": "6",
                "explanation": "[4,-1,2,1] has the largest sum = 6.",
            }
        ],
        "external_link": "https://leetcode.com/problems/maximum-subarray/",
    },
    {
        "leetcode_id": 70,
        "title": "Climbing Stairs",
        "difficulty": "Easy",
        "category": "Dynamic Programming",
        "description": "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
        "solution": """def climbStairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b""",
        "solution_approach": "Dynamic Programming (Fibonacci)",
        "examples": [
            {"input": "n = 2", "output": "2", "explanation": "1+1 or 2 steps."},
            {"input": "n = 3", "output": "3", "explanation": "1+1+1, 1+2, or 2+1."},
        ],
        "external_link": "https://leetcode.com/problems/climbing-stairs/",
    },
    {
        "leetcode_id": 56,
        "title": "Merge Intervals",
        "difficulty": "Medium",
        "category": "Array",
        "description": "Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.",
        "solution": """def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged""",
        "solution_approach": "Sorting",
        "examples": [
            {
                "input": "intervals = [[1,3],[2,6],[8,10],[15,18]]",
                "output": "[[1,6],[8,10],[15,18]]",
                "explanation": "Intervals [1,3] and [2,6] overlap.",
            }
        ],
        "external_link": "https://leetcode.com/problems/merge-intervals/",
    },
    {
        "leetcode_id": 15,
        "title": "3Sum",
        "difficulty": "Medium",
        "category": "Two Pointers",
        "description": "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.",
        "solution": """def threeSum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result""",
        "solution_approach": "Two Pointers with Sorting",
        "examples": [
            {
                "input": "nums = [-1,0,1,2,-1,-4]",
                "output": "[[-1,-1,2],[-1,0,1]]",
                "explanation": "Three triplets that sum to zero.",
            }
        ],
        "external_link": "https://leetcode.com/problems/3sum/",
    },
    {
        "leetcode_id": 11,
        "title": "Container With Most Water",
        "difficulty": "Medium",
        "category": "Two Pointers",
        "description": "You are given an integer array height of length n. Find two lines that together with the x-axis form a container, such that the container contains the most water.",
        "solution": """def maxArea(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water""",
        "solution_approach": "Two Pointers",
        "examples": [
            {
                "input": "height = [1,8,6,2,5,4,8,3,7]",
                "output": "49",
                "explanation": "Lines at index 1 and 8 form container with max water.",
            }
        ],
        "external_link": "https://leetcode.com/problems/container-with-most-water/",
    },
    {
        "leetcode_id": 33,
        "title": "Search in Rotated Sorted Array",
        "difficulty": "Medium",
        "category": "Binary Search",
        "description": "Given an integer array nums sorted in ascending order with distinct values, which is then rotated at an unknown pivot, search for target and return its index, or -1 if not found.",
        "solution": """def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1""",
        "solution_approach": "Modified Binary Search",
        "examples": [
            {
                "input": "nums = [4,5,6,7,0,1,2], target = 0",
                "output": "4",
                "explanation": "0 is at index 4.",
            }
        ],
        "external_link": "https://leetcode.com/problems/search-in-rotated-sorted-array/",
    },
    {
        "leetcode_id": 200,
        "title": "Number of Islands",
        "difficulty": "Medium",
        "category": "Graph",
        "description": "Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), return the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.",
        "solution": """def numIslands(grid):
    if not grid:
        return 0
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(grid, i, j)
                count += 1
    return count

def dfs(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1':
        return
    grid[i][j] = '0'
    dfs(grid, i+1, j)
    dfs(grid, i-1, j)
    dfs(grid, i, j+1)
    dfs(grid, i, j-1)""",
        "solution_approach": "DFS Flood Fill",
        "examples": [
            {
                "input": 'grid = [["1","1","0"],["1","1","0"],["0","0","1"]]',
                "output": "2",
                "explanation": "Two islands found.",
            }
        ],
        "external_link": "https://leetcode.com/problems/number-of-islands/",
    },
    {
        "leetcode_id": 206,
        "title": "Reverse Linked List",
        "difficulty": "Easy",
        "category": "Linked List",
        "description": "Given the head of a singly linked list, reverse the list, and return the reversed list.",
        "solution": """def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev""",
        "solution_approach": "Iterative Pointer Reversal",
        "examples": [
            {
                "input": "head = [1,2,3,4,5]",
                "output": "[5,4,3,2,1]",
                "explanation": "List reversed.",
            }
        ],
        "external_link": "https://leetcode.com/problems/reverse-linked-list/",
    },
    {
        "leetcode_id": 141,
        "title": "Linked List Cycle",
        "difficulty": "Easy",
        "category": "Linked List",
        "description": "Given head, the head of a linked list, determine if the linked list has a cycle in it.",
        "solution": """def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False""",
        "solution_approach": "Floyd's Cycle Detection (Tortoise and Hare)",
        "examples": [
            {
                "input": "head = [3,2,0,-4], pos = 1",
                "output": "true",
                "explanation": "There is a cycle where tail connects to index 1.",
            }
        ],
        "external_link": "https://leetcode.com/problems/linked-list-cycle/",
    },
    {
        "leetcode_id": 102,
        "title": "Binary Tree Level Order Traversal",
        "difficulty": "Medium",
        "category": "Tree",
        "description": "Given the root of a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level).",
        "solution": """from collections import deque

def levelOrder(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result""",
        "solution_approach": "BFS with Queue",
        "examples": [
            {
                "input": "root = [3,9,20,null,null,15,7]",
                "output": "[[3],[9,20],[15,7]]",
                "explanation": "Level-by-level traversal.",
            }
        ],
        "external_link": "https://leetcode.com/problems/binary-tree-level-order-traversal/",
    },
    {
        "leetcode_id": 104,
        "title": "Maximum Depth of Binary Tree",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "Given the root of a binary tree, return its maximum depth. A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.",
        "solution": """def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))""",
        "solution_approach": "Recursive DFS",
        "examples": [
            {
                "input": "root = [3,9,20,null,null,15,7]",
                "output": "3",
                "explanation": "3 levels deep.",
            }
        ],
        "external_link": "https://leetcode.com/problems/maximum-depth-of-binary-tree/",
    },
    {
        "leetcode_id": 226,
        "title": "Invert Binary Tree",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "Given the root of a binary tree, invert the tree, and return its root.",
        "solution": """def invertTree(root):
    if not root:
        return None
    root.left, root.right = root.right, root.left
    invertTree(root.left)
    invertTree(root.right)
    return root""",
        "solution_approach": "Recursive Swap",
        "examples": [
            {
                "input": "root = [4,2,7,1,3,6,9]",
                "output": "[4,7,2,9,6,3,1]",
                "explanation": "Tree mirrored.",
            }
        ],
        "external_link": "https://leetcode.com/problems/invert-binary-tree/",
    },
    {
        "leetcode_id": 238,
        "title": "Product of Array Except Self",
        "difficulty": "Medium",
        "category": "Array",
        "description": "Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]. You must solve it in O(n) time without using division.",
        "solution": """def productExceptSelf(nums):
    n = len(nums)
    answer = [1] * n
    prefix = 1
    for i in range(n):
        answer[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix
        suffix *= nums[i]
    return answer""",
        "solution_approach": "Prefix and Suffix Products",
        "examples": [
            {
                "input": "nums = [1,2,3,4]",
                "output": "[24,12,8,6]",
                "explanation": "Each element is product of all others.",
            }
        ],
        "external_link": "https://leetcode.com/problems/product-of-array-except-self/",
    },
    {
        "leetcode_id": 49,
        "title": "Group Anagrams",
        "difficulty": "Medium",
        "category": "Hash Table",
        "description": "Given an array of strings strs, group the anagrams together. You can return the answer in any order.",
        "solution": """from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())""",
        "solution_approach": "Hash Map with Sorted Key",
        "examples": [
            {
                "input": 'strs = ["eat","tea","tan","ate","nat","bat"]',
                "output": '[["bat"],["nat","tan"],["ate","eat","tea"]]',
                "explanation": "Anagrams grouped together.",
            }
        ],
        "external_link": "https://leetcode.com/problems/group-anagrams/",
    },
    {
        "leetcode_id": 42,
        "title": "Trapping Rain Water",
        "difficulty": "Hard",
        "category": "Two Pointers",
        "description": "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.",
        "solution": """def trap(height):
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    result = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            result += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            result += right_max - height[right]
    return result""",
        "solution_approach": "Two Pointers",
        "examples": [
            {
                "input": "height = [0,1,0,2,1,0,1,3,2,1,2,1]",
                "output": "6",
                "explanation": "6 units of water trapped.",
            }
        ],
        "external_link": "https://leetcode.com/problems/trapping-rain-water/",
    },
    {
        "leetcode_id": 23,
        "title": "Merge k Sorted Lists",
        "difficulty": "Hard",
        "category": "Heap",
        "description": "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
        "solution": """import heapq

def mergeKLists(lists):
    heap = []
    for i, l in enumerate(lists):
        if l:
            heapq.heappush(heap, (l.val, i, l))
    dummy = ListNode()
    curr = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next""",
        "solution_approach": "Min Heap",
        "examples": [
            {
                "input": "lists = [[1,4,5],[1,3,4],[2,6]]",
                "output": "[1,1,2,3,4,4,5,6]",
                "explanation": "All lists merged sorted.",
            }
        ],
        "external_link": "https://leetcode.com/problems/merge-k-sorted-lists/",
    },
    {
        "leetcode_id": 76,
        "title": "Minimum Window Substring",
        "difficulty": "Hard",
        "category": "Sliding Window",
        "description": "Given two strings s and t, return the minimum window substring of s such that every character in t (including duplicates) is included in the window.",
        "solution": """from collections import Counter

def minWindow(s, t):
    if not t or not s:
        return ""
    t_count = Counter(t)
    required = len(t_count)
    left = right = formed = 0
    window_counts = {}
    ans = float("inf"), None, None
    while right < len(s):
        c = s[right]
        window_counts[c] = window_counts.get(c, 0) + 1
        if c in t_count and window_counts[c] == t_count[c]:
            formed += 1
        while left <= right and formed == required:
            c = s[left]
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            window_counts[c] -= 1
            if c in t_count and window_counts[c] < t_count[c]:
                formed -= 1
            left += 1
        right += 1
    return "" if ans[0] == float("inf") else s[ans[1]:ans[2]+1]""",
        "solution_approach": "Sliding Window with Hash Map",
        "examples": [
            {
                "input": 's = "ADOBECODEBANC", t = "ABC"',
                "output": '"BANC"',
                "explanation": "Minimum window containing all characters of t.",
            }
        ],
        "external_link": "https://leetcode.com/problems/minimum-window-substring/",
    },
    {
        "leetcode_id": 152,
        "title": "Maximum Product Subarray",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given an integer array nums, find a subarray that has the largest product, and return the product.",
        "solution": """def maxProduct(nums):
    result = max(nums)
    cur_min, cur_max = 1, 1
    for n in nums:
        vals = (n, n * cur_max, n * cur_min)
        cur_max, cur_min = max(vals), min(vals)
        result = max(result, cur_max)
    return result""",
        "solution_approach": "Dynamic Programming with Min/Max Tracking",
        "examples": [
            {
                "input": "nums = [2,3,-2,4]",
                "output": "6",
                "explanation": "[2,3] has largest product = 6.",
            }
        ],
        "external_link": "https://leetcode.com/problems/maximum-product-subarray/",
    },
    {
        "leetcode_id": 153,
        "title": "Find Minimum in Rotated Sorted Array",
        "difficulty": "Medium",
        "category": "Binary Search",
        "description": "Given the sorted rotated array nums of unique elements, return the minimum element of this array. You must write an algorithm that runs in O(log n) time.",
        "solution": """def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]""",
        "solution_approach": "Binary Search",
        "examples": [
            {
                "input": "nums = [3,4,5,1,2]",
                "output": "1",
                "explanation": "Array was rotated 3 times, minimum is 1.",
            }
        ],
        "external_link": "https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/",
    },
    {
        "leetcode_id": 198,
        "title": "House Robber",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given an integer array nums representing money at each house, return the maximum amount you can rob tonight without alerting the police (can't rob two adjacent houses).",
        "solution": """def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev1, prev2 = 0, 0
    for num in nums:
        temp = prev1
        prev1 = max(prev2 + num, prev1)
        prev2 = temp
    return prev1""",
        "solution_approach": "Dynamic Programming",
        "examples": [
            {
                "input": "nums = [1,2,3,1]",
                "output": "4",
                "explanation": "Rob house 1 (1) + house 3 (3) = 4.",
            }
        ],
        "external_link": "https://leetcode.com/problems/house-robber/",
    },
    {
        "leetcode_id": 322,
        "title": "Coin Change",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given an integer array coins representing coin denominations and an integer amount, return the fewest number of coins needed to make up that amount. If that amount cannot be made, return -1.",
        "solution": """def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1""",
        "solution_approach": "Dynamic Programming (Bottom-Up)",
        "examples": [
            {
                "input": "coins = [1,2,5], amount = 11",
                "output": "3",
                "explanation": "11 = 5 + 5 + 1.",
            }
        ],
        "external_link": "https://leetcode.com/problems/coin-change/",
    },
    {
        "leetcode_id": 300,
        "title": "Longest Increasing Subsequence",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given an integer array nums, return the length of the longest strictly increasing subsequence.",
        "solution": """def lengthOfLIS(nums):
    from bisect import bisect_left
    sub = []
    for num in nums:
        pos = bisect_left(sub, num)
        if pos == len(sub):
            sub.append(num)
        else:
            sub[pos] = num
    return len(sub)""",
        "solution_approach": "Binary Search with Patience Sorting",
        "examples": [
            {
                "input": "nums = [10,9,2,5,3,7,101,18]",
                "output": "4",
                "explanation": "LIS is [2,3,7,101].",
            }
        ],
        "external_link": "https://leetcode.com/problems/longest-increasing-subsequence/",
    },
    {
        "leetcode_id": 128,
        "title": "Longest Consecutive Sequence",
        "difficulty": "Medium",
        "category": "Hash Table",
        "description": "Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence. You must write an algorithm that runs in O(n) time.",
        "solution": """def longestConsecutive(nums):
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:
            length = 1
            while num + length in num_set:
                length += 1
            longest = max(longest, length)
    return longest""",
        "solution_approach": "Hash Set",
        "examples": [
            {
                "input": "nums = [100,4,200,1,3,2]",
                "output": "4",
                "explanation": "Sequence [1,2,3,4] has length 4.",
            }
        ],
        "external_link": "https://leetcode.com/problems/longest-consecutive-sequence/",
    },
    {
        "leetcode_id": 242,
        "title": "Valid Anagram",
        "difficulty": "Easy",
        "category": "Hash Table",
        "description": "Given two strings s and t, return true if t is an anagram of s, and false otherwise. An anagram is a word or phrase formed by rearranging the letters of a different word or phrase.",
        "solution": """def isAnagram(s, t):
    if len(s) != len(t):
        return False
    count = [0] * 26
    for i in range(len(s)):
        count[ord(s[i]) - ord('a')] += 1
        count[ord(t[i]) - ord('a')] -= 1
    return all(c == 0 for c in count)""",
        "solution_approach": "Hash Map with Character Count",
        "examples": [
            {
                "input": 's = "anagram", t = "nagaram"',
                "output": "true",
                "explanation": "Both strings contain the same letters with same frequencies.",
            },
            {
                "input": 's = "rat", t = "car"',
                "output": "false",
                "explanation": "Different letters.",
            },
        ],
        "external_link": "https://leetcode.com/problems/valid-anagram/",
    },
    {
        "leetcode_id": 101,
        "title": "Symmetric Tree",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).",
        "solution": """def isSymmetric(root):
    def isMirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return t1.val == t2.val and isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left)
    return isMirror(root, root)""",
        "solution_approach": "Recursive Tree Comparison",
        "examples": [
            {
                "input": "root = [1,2,2,3,4,4,3]",
                "output": "true",
                "explanation": "Tree is symmetric.",
            },
            {
                "input": "root = [1,2,2,null,3,null,3]",
                "output": "false",
                "explanation": "Tree is not symmetric.",
            },
        ],
        "external_link": "https://leetcode.com/problems/symmetric-tree/",
    },
    {
        "leetcode_id": 94,
        "title": "Binary Tree Inorder Traversal",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "Given the root of a binary tree, return the inorder traversal of its nodes' values (left, root, right).",
        "solution": """def inorderTraversal(root):
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)
    dfs(root)
    return result""",
        "solution_approach": "Recursive DFS",
        "examples": [
            {
                "input": "root = [1,null,2,3]",
                "output": "[1,3,2]",
                "explanation": "Inorder traversal visits left, root, right.",
            }
        ],
        "external_link": "https://leetcode.com/problems/binary-tree-inorder-traversal/",
    },
    {
        "leetcode_id": 98,
        "title": "Validate Binary Search Tree",
        "difficulty": "Medium",
        "category": "Tree",
        "description": "Given the root of a binary tree, determine if it is a valid binary search tree. A BST is defined as left subtree < node < right subtree, and recursively.",
        "solution": """def isValidBST(root):
    def validate(node, low, high):
        if not node:
            return True
        if node.val <= low or node.val >= high:
            return False
        return validate(node.left, low, node.val) and validate(node.right, node.val, high)
    return validate(root, float('-inf'), float('inf'))""",
        "solution_approach": "Recursive Validation with Bounds",
        "examples": [
            {"input": "root = [2,1,3]", "output": "true", "explanation": "Valid BST."},
            {
                "input": "root = [5,1,4,null,null,3,6]",
                "output": "false",
                "explanation": "Right child of 5 is 4, which is less than 5.",
            },
        ],
        "external_link": "https://leetcode.com/problems/validate-binary-search-tree/",
    },
    {
        "leetcode_id": 112,
        "title": "Path Sum",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "Given the root of a binary tree and an integer targetSum, return true if there exists a root-to-leaf path such that the sum of node values equals targetSum.",
        "solution": """def hasPathSum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    return hasPathSum(root.left, targetSum - root.val) or hasPathSum(root.right, targetSum - root.val)""",
        "solution_approach": "Recursive DFS with Accumulation",
        "examples": [
            {
                "input": "root = [5,4,8,11,null,13,4,7,2], targetSum = 22",
                "output": "true",
                "explanation": "Path 5->4->11->2 sums to 22.",
            }
        ],
        "external_link": "https://leetcode.com/problems/path-sum/",
    },
    {
        "leetcode_id": 105,
        "title": "Construct Binary Tree from Preorder and Inorder",
        "difficulty": "Medium",
        "category": "Tree",
        "description": "Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal, construct and return the binary tree.",
        "solution": """def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])
    return root""",
        "solution_approach": "Recursion with Index Tracking",
        "examples": [
            {
                "input": "preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]",
                "output": "[3,9,20,null,null,15,7]",
                "explanation": "Tree reconstructed from traversals.",
            }
        ],
        "external_link": "https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/",
    },
    {
        "leetcode_id": 79,
        "title": "Word Search",
        "difficulty": "Medium",
        "category": "Backtracking",
        "description": "Given an m x n board and a word, find if the word exists in the board. The word can be constructed from letters of adjacent cells (horizontally or vertically).",
        "solution": """def exist(board, word):
    rows, cols = len(board), len(board[0])
    def dfs(r, c, i):
        if i == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[i]:
            return False
        board[r][c] = '#'
        found = dfs(r+1, c, i+1) or dfs(r-1, c, i+1) or dfs(r, c+1, i+1) or dfs(r, c-1, i+1)
        board[r][c] = word[i]
        return found
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False""",
        "solution_approach": "DFS Backtracking",
        "examples": [
            {
                "input": 'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"',
                "output": "true",
                "explanation": "Word found in the board.",
            }
        ],
        "external_link": "https://leetcode.com/problems/word-search/",
    },
    {
        "leetcode_id": 78,
        "title": "Subsets",
        "difficulty": "Medium",
        "category": "Backtracking",
        "description": "Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets.",
        "solution": """def subsets(nums):
    result = [[]]
    for num in nums:
        result += [curr + [num] for curr in result]
    return result""",
        "solution_approach": "Iterative Subset Generation",
        "examples": [
            {
                "input": "nums = [1,2,3]",
                "output": "[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]",
                "explanation": "All possible subsets.",
            }
        ],
        "external_link": "https://leetcode.com/problems/subsets/",
    },
    {
        "leetcode_id": 46,
        "title": "Permutations",
        "difficulty": "Medium",
        "category": "Backtracking",
        "description": "Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.",
        "solution": """def permute(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return result""",
        "solution_approach": "Backtracking with Used Array",
        "examples": [
            {
                "input": "nums = [1,2,3]",
                "output": "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]",
                "explanation": "All permutations of the array.",
            }
        ],
        "external_link": "https://leetcode.com/problems/permutations/",
    },
    {
        "leetcode_id": 57,
        "title": "Insert Interval",
        "difficulty": "Medium",
        "category": "Array",
        "description": "You are given an array of non-overlapping intervals intervals and a new interval newInterval. Insert newInterval into intervals such that intervals are still non-overlapping and merged if necessary.",
        "solution": """def insert(intervals, newInterval):
    result = []
    i = 0
    n = len(intervals)
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)
    while i < n:
        result.append(intervals[i])
        i += 1
    return result""",
        "solution_approach": "Merge Overlapping Intervals",
        "examples": [
            {
                "input": "intervals = [[1,3],[6,9]], newInterval = [2,5]",
                "output": "[[1,5],[6,9]]",
                "explanation": "New interval merges with [1,3].",
            }
        ],
        "external_link": "https://leetcode.com/problems/insert-interval/",
    },
    {
        "leetcode_id": 54,
        "title": "Spiral Matrix",
        "difficulty": "Medium",
        "category": "Array",
        "description": "Given an m x n matrix, return all elements of the matrix in spiral order.",
        "solution": """def spiralOrder(matrix):
    if not matrix:
        return []
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result""",
        "solution_approach": "Boundary Tracking",
        "examples": [
            {
                "input": "matrix = [[1,2,3],[4,5,6],[7,8,9]]",
                "output": "[1,2,3,6,9,8,7,4,5]",
                "explanation": "Elements in spiral order.",
            }
        ],
        "external_link": "https://leetcode.com/problems/spiral-matrix/",
    },
    {
        "leetcode_id": 55,
        "title": "Jump Game",
        "difficulty": "Medium",
        "category": "Greedy",
        "description": "Given an array of non-negative integers nums, you are initially positioned at the first index. Determine if you can reach the last index.",
        "solution": """def canJump(nums):
    farthest = 0
    for i, jump in enumerate(nums):
        if i > farthest:
            return False
        farthest = max(farthest, i + jump)
        if farthest >= len(nums) - 1:
            return True
    return True""",
        "solution_approach": "Greedy Reach Tracking",
        "examples": [
            {
                "input": "nums = [2,3,1,1,4]",
                "output": "true",
                "explanation": "Can reach end with jumps.",
            },
            {
                "input": "nums = [3,2,1,0,4]",
                "output": "false",
                "explanation": "Stuck at index 3.",
            },
        ],
        "external_link": "https://leetcode.com/problems/jump-game/",
    },
    {
        "leetcode_id": 88,
        "title": "Merge Sorted Array",
        "difficulty": "Easy",
        "category": "Array",
        "description": "You are given two integer arrays nums1 and nums2, sorted in non-decreasing order. Merge nums2 into nums1 as one sorted array.",
        "solution": """def merge(nums1, m, nums2, n):
    p1, p2 = m - 1, n - 1
    p = m + n - 1
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    nums1[:p2+1] = nums2[:p2+1]""",
        "solution_approach": "Two Pointers from End",
        "examples": [
            {
                "input": "nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3",
                "output": "[1,2,2,3,5,6]",
                "explanation": "Merged sorted array.",
            }
        ],
        "external_link": "https://leetcode.com/problems/merge-sorted-array/",
    },
    {
        "leetcode_id": 278,
        "title": "First Bad Version",
        "difficulty": "Easy",
        "category": "Binary Search",
        "description": "You are a product manager, and currently leading a team to develop a new product. All versions after a bad version are also bad. Find the first bad version.",
        "solution": """def firstBadVersion(n):
    left, right = 1, n
    while left < right:
        mid = left + (right - left) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    return left""",
        "solution_approach": "Binary Search",
        "examples": [
            {
                "input": "n = 5, bad = 4",
                "output": "4",
                "explanation": "First bad version is 4.",
            }
        ],
        "external_link": "https://leetcode.com/problems/first-bad-version/",
    },
    {
        "leetcode_id": 69,
        "title": "Sqrt(x)",
        "difficulty": "Easy",
        "category": "Binary Search",
        "description": "Given a non-negative integer x, compute and return the square root of x rounded down to the nearest integer.",
        "solution": """def mySqrt(x):
    if x < 2:
        return x
    left, right = 1, x // 2
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return right""",
        "solution_approach": "Binary Search",
        "examples": [
            {
                "input": "x = 8",
                "output": "2",
                "explanation": "sqrt(8) ≈ 2.828, floor is 2.",
            }
        ],
        "external_link": "https://leetcode.com/problems/sqrtx/",
    },
    {
        "leetcode_id": 167,
        "title": "Two Sum II - Input Array Is Sorted",
        "difficulty": "Medium",
        "category": "Two Pointers",
        "description": "Given a 1-indexed array of integers numbers sorted in non-decreasing order, find two numbers such that they add up to a target number.",
        "solution": """def twoSum(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []""",
        "solution_approach": "Two Pointers",
        "examples": [
            {
                "input": "numbers = [2,7,11,15], target = 9",
                "output": "[1,2]",
                "explanation": "numbers[0] + numbers[1] = 2 + 7 = 9.",
            }
        ],
        "external_link": "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/",
    },
    {
        "leetcode_id": 155,
        "title": "Min Stack",
        "difficulty": "Medium",
        "category": "Stack",
        "description": "Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.",
        "solution": """class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
    def top(self):
        return self.stack[-1]
    def getMin(self):
        return self.min_stack[-1]""",
        "solution_approach": "Auxiliary Min Stack",
        "examples": [
            {
                "input": 'ops = ["MinStack","push","push","push","getMin","pop","getMin"]',
                "output": "[null,null,null,null,1,null,2]",
                "explanation": "Stack operations with min tracking.",
            }
        ],
        "external_link": "https://leetcode.com/problems/min-stack/",
    },
    {
        "leetcode_id": 139,
        "title": "Word Break",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.",
        "solution": """def wordBreak(s, wordDict):
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]""",
        "solution_approach": "Dynamic Programming",
        "examples": [
            {
                "input": 's = "leetcode", wordDict = ["leet","code"]',
                "output": "true",
                "explanation": "Can be segmented as leet + code.",
            }
        ],
        "external_link": "https://leetcode.com/problems/word-break/",
    },
    {
        "leetcode_id": 134,
        "title": "Gas Station",
        "difficulty": "Medium",
        "category": "Greedy",
        "description": "There are n gas stations along a circular route. Given two integer arrays gas and cost, return the starting gas station index if you can travel around the circuit once, otherwise return -1.",
        "solution": """def gasStation(gas, cost):
    total_tank, curr_tank = 0, 0
    start = 0
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        curr_tank += gas[i] - cost[i]
        if curr_tank < 0:
            start = i + 1
            curr_tank = 0
    return start if total_tank >= 0 else -1""",
        "solution_approach": "Greedy with Tank Tracking",
        "examples": [
            {
                "input": "gas = [1,2,3,4,5], cost = [3,4,5,1,2]",
                "output": "3",
                "explanation": "Start at station 3, can complete circuit.",
            }
        ],
        "external_link": "https://leetcode.com/problems/gas-station/",
    },
    {
        "leetcode_id": 62,
        "title": "Unique Paths",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "There is a robot on an m x n grid. The robot starts at the top-left corner (0,0). Find the number of different paths to reach the bottom-right corner.",
        "solution": """def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]""",
        "solution_approach": "Dynamic Programming",
        "examples": [
            {
                "input": "m = 3, n = 7",
                "output": "28",
                "explanation": "28 unique paths from top-left to bottom-right.",
            }
        ],
        "external_link": "https://leetcode.com/problems/unique-paths/",
    },
    {
        "leetcode_id": 64,
        "title": "Minimum Path Sum",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given a m x n grid filled with non-negative numbers, find a path from top-left to bottom-right that minimizes the sum of the numbers along the path.",
        "solution": """def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                grid[i][j] += grid[i][j-1]
            elif j == 0:
                grid[i][j] += grid[i-1][j]
            else:
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[m-1][n-1]""",
        "solution_approach": "Dynamic Programming (In-Place)",
        "examples": [
            {
                "input": "grid = [[1,3,1],[1,5,1],[4,2,1]]",
                "output": "7",
                "explanation": "Path 1→3→1→1→1 sums to 7.",
            }
        ],
        "external_link": "https://leetcode.com/problems/minimum-path-sum/",
    },
    {
        "leetcode_id": 148,
        "title": "Sort List",
        "difficulty": "Medium",
        "category": "Linked List",
        "description": "Given the head of a linked list, return the list after sorting it in ascending order.",
        "solution": """def sortList(head):
    if not head or not head.next:
        return head
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None
    left = sortList(head)
    right = sortList(mid)
    return merge(left, right)

def merge(l1, l2):
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next""",
        "solution_approach": "Merge Sort",
        "examples": [
            {
                "input": "head = [4,2,1,3]",
                "output": "[1,2,3,4]",
                "explanation": "List sorted in ascending order.",
            }
        ],
        "external_link": "https://leetcode.com/problems/sort-list/",
    },
    {
        "leetcode_id": 146,
        "title": "LRU Cache",
        "difficulty": "Medium",
        "category": "Hash Table",
        "description": "Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.",
        "solution": """from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)""",
        "solution_approach": "Ordered Dictionary",
        "examples": [
            {
                "input": 'ops = ["LRUCache","put","put","get","put","get","put","get","get","get"]',
                "output": "[null,null,null,1,null,-1,null,-1,3,4]",
                "explanation": "LRU cache operations.",
            }
        ],
        "external_link": "https://leetcode.com/problems/lru-cache/",
    },
    {
        "leetcode_id": 207,
        "title": "Course Schedule",
        "difficulty": "Medium",
        "category": "Graph",
        "description": "There are numCourses courses you have to take. Return whether it is possible to finish all courses given prerequisites as pairs [ai, bi] meaning you must take bi before ai.",
        "solution": """def canFinish(numCourses, prerequisites):
    from collections import deque, defaultdict
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    for dest, src in prerequisites:
        graph[src].append(dest)
        in_degree[dest] += 1
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0
    while queue:
        node = queue.popleft()
        count += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return count == numCourses""",
        "solution_approach": "Topological Sort (Kahn's Algorithm)",
        "examples": [
            {
                "input": "numCourses = 2, prerequisites = [[1,0]]",
                "output": "true",
                "explanation": "Can finish all courses.",
            }
        ],
        "external_link": "https://leetcode.com/problems/course-schedule/",
    },
    {
        "leetcode_id": 210,
        "title": "Course Schedule II",
        "difficulty": "Medium",
        "category": "Graph",
        "description": "Find one of the possible orders to take all courses given prerequisites.",
        "solution": """def findOrder(numCourses, prerequisites):
    from collections import deque, defaultdict
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    for dest, src in prerequisites:
        graph[src].append(dest)
        in_degree[dest] += 1
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result if len(result) == numCourses else []""",
        "solution_approach": "Topological Sort",
        "examples": [
            {
                "input": "numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]",
                "output": "[0,1,2,3]",
                "explanation": "Valid course order.",
            }
        ],
        "external_link": "https://leetcode.com/problems/course-schedule-ii/",
    },
    {
        "leetcode_id": 236,
        "title": "Lowest Common Ancestor of a Binary Tree",
        "difficulty": "Medium",
        "category": "Tree",
        "description": "Given a binary tree root and two nodes p and q, find the lowest common ancestor (LCA) of p and q.",
        "solution": """def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left or right""",
        "solution_approach": "Recursive Tree Traversal",
        "examples": [
            {
                "input": "root = [3,5,1,6,2,0,7,null,null,4,1], p = 5, q = 1",
                "output": "3",
                "explanation": "LCA of nodes 5 and 1 is node 3.",
            }
        ],
        "external_link": "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/",
    },
    {
        "leetcode_id": 347,
        "title": "Top K Frequent Elements",
        "difficulty": "Medium",
        "category": "Hash Table",
        "description": "Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.",
        "solution": """from collections import Counter
import heapq

def topKFrequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)""",
        "solution_approach": "Counter with Heap",
        "examples": [
            {
                "input": "nums = [1,1,1,2,2,3], k = 2",
                "output": "[1,2]",
                "explanation": "Most frequent elements are 1 and 2.",
            }
        ],
        "external_link": "https://leetcode.com/problems/top-k-frequent-elements/",
    },
    {
        "leetcode_id": 394,
        "title": "Decode String",
        "difficulty": "Medium",
        "category": "Stack",
        "description": "Given an encoded string, return its decoded string. The encoding rule is: k[encoded_string] where encoded_string is repeated k times.",
        "solution": """def decodeString(s):
    stack = []
    cur_num = 0
    cur_str = []
    for char in s:
        if char == '[':
            stack.append((cur_num, cur_str))
            cur_num = 0
            cur_str = []
        elif char == ']':
            num, prev_str = stack.pop()
            cur_str = prev_str + cur_str * num
        elif char.isdigit():
            cur_num = cur_num * 10 + int(char)
        else:
            cur_str.append(char)
    return ''.join(cur_str)""",
        "solution_approach": "Stack with Nesting",
        "examples": [
            {
                "input": 's = "3[a2[c]]"',
                "output": '"accaccacc"',
                "explanation": "3 times 'acc'.",
            }
        ],
        "external_link": "https://leetcode.com/problems/decode-string/",
    },
    {
        "leetcode_id": 437,
        "title": "Path Sum III",
        "difficulty": "Medium",
        "category": "Tree",
        "description": "Given the root of a binary tree and an integer targetSum, return the number of paths that sum to targetSum. Paths can start and end anywhere in the tree.",
        "solution": """def pathSum(root, targetSum):
    def dfs(node, curr_sum):
        if not node:
            return 0
        curr_sum += node.val
        count = prefix_sum.get(curr_sum - targetSum, 0)
        prefix_sum[curr_sum] = prefix_sum.get(curr_sum, 0) + 1
        count += dfs(node.left, curr_sum)
        count += dfs(node.right, curr_sum)
        prefix_sum[curr_sum] -= 1
        return count
    prefix_sum = {}
    return dfs(root, 0)""",
        "solution_approach": "Prefix Sum with Hash Map",
        "examples": [
            {
                "input": "root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8",
                "output": "3",
                "explanation": "Three paths sum to 8.",
            }
        ],
        "external_link": "https://leetcode.com/problems/path-sum-iii/",
    },
    {
        "leetcode_id": 448,
        "title": "Find All Numbers Disappeared in an Array",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers in the range [1, n] that do not appear in nums.",
        "solution": """def findDisappearedNumbers(nums):
    n = len(nums)
    for num in nums:
        index = abs(num) - 1
        nums[index] = -abs(nums[index])
    return [i + 1 for i, x in enumerate(nums) if x > 0]""",
        "solution_approach": "In-place Marking",
        "examples": [
            {
                "input": "nums = [4,3,2,7,8,2,3,1]",
                "output": "[5,6]",
                "explanation": "5 and 6 are missing from the array.",
            }
        ],
        "external_link": "https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/",
    },
    {
        "leetcode_id": 461,
        "title": "Hamming Distance",
        "difficulty": "Easy",
        "category": "Bit Manipulation",
        "description": "Given two integers x and y, return the Hamming distance between their binary representations.",
        "solution": """def hammingDistance(x, y):
    return bin(x ^ y).count('1')""",
        "solution_approach": "XOR and Bit Count",
        "examples": [
            {
                "input": "x = 1, y = 4",
                "output": "2",
                "explanation": "1 (01) and 4 (100) differ in 2 bits.",
            }
        ],
        "external_link": "https://leetcode.com/problems/hamming-distance/",
    },
    {
        "leetcode_id": 543,
        "title": "Diameter of Binary Tree",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "Given the root of a binary tree, return the length of the diameter of the tree. The diameter is the longest path between any two nodes in the tree.",
        "solution": """def diameterOfBinaryTree(root):
    max_diameter = 0
    def depth(node):
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        nonlocal max_diameter
        max_diameter = max(max_diameter, left + right)
        return 1 + max(left, right)
    depth(root)
    return max_diameter""",
        "solution_approach": "DFS with Diameter Tracking",
        "examples": [
            {
                "input": "root = [1,2,3,4,5]",
                "output": "3",
                "explanation": "Diameter: 4->2->1->3 or 4->2->1->3.",
            }
        ],
        "external_link": "https://leetcode.com/problems/diameter-of-binary-tree/",
    },
    {
        "leetcode_id": 617,
        "title": "Merge Two Binary Trees",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "You are given two binary trees root1 and root2. Imagine that when you put one of them to cover the other, some nodes of the two trees overlap while others are not. Merge the two trees into a single binary tree.",
        "solution": """def mergeTrees(root1, root2):
    if not root1:
        return root2
    if not root2:
        return root1
    root1.val += root2.val
    root1.left = mergeTrees(root1.left, root2.left)
    root1.right = mergeTrees(root1.right, root2.right)
    return root1""",
        "solution_approach": "Recursive Tree Merging",
        "examples": [
            {
                "input": "root1 = [1,3,2,5], root2 = [2,1,3,null,4,7]",
                "output": "[3,4,5,5,4,null,7]",
                "explanation": "Trees merged together.",
            }
        ],
        "external_link": "https://leetcode.com/problems/merge-two-binary-trees/",
    },
    {
        "leetcode_id": 704,
        "title": "Binary Search",
        "difficulty": "Easy",
        "category": "Binary Search",
        "description": "Given a sorted array of integers nums and an integer target, search for target in nums. If target exists, return its index. Otherwise, return -1.",
        "solution": """def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
        "solution_approach": "Standard Binary Search",
        "examples": [
            {
                "input": "nums = [-1,0,3,5,9,12], target = 9",
                "output": "4",
                "explanation": "9 is at index 4.",
            }
        ],
        "external_link": "https://leetcode.com/problems/binary-search/",
    },
    {
        "leetcode_id": 733,
        "title": "Flood Fill",
        "difficulty": "Easy",
        "category": "DFS",
        "description": "An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image. Starting from the pixel image[sr][sc], flood fill the image.",
        "solution": """def floodFill(image, sr, sc, newColor):
    from collections import deque
    rows, cols = len(image), len(image[0])
    original = image[sr][sc]
    if original == newColor:
        return image
    queue = deque([(sr, sc)])
    image[sr][sc] = newColor
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and image[nr][nc] == original:
                image[nr][nc] = newColor
                queue.append((nr, nc))
    return image""",
        "solution_approach": "BFS Flood Fill",
        "examples": [
            {
                "input": "image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2",
                "output": "[[2,2,2],[2,2,0],[2,0,1]]",
                "explanation": "Center pixel and connected pixels changed to 2.",
            }
        ],
        "external_link": "https://leetcode.com/problems/flood-fill/",
    },
    {
        "leetcode_id": 763,
        "title": "Partition Labels",
        "difficulty": "Medium",
        "category": "Greedy",
        "description": "Given a string s, partition the string into as many parts as possible such that each letter appears in at most one part. Return an array representing the size of each part.",
        "solution": """def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    result = []
    start, end = 0, 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result""",
        "solution_approach": "Last Occurrence Tracking",
        "examples": [
            {
                "input": "s = ababcbacadefegdehijhklij",
                "output": "[9,7,8]",
                "explanation": "Partitions based on character ranges.",
            }
        ],
        "external_link": "https://leetcode.com/problems/partition-labels/",
    },
    {
        "leetcode_id": 841,
        "title": "Keys and Rooms",
        "difficulty": "Medium",
        "category": "DFS",
        "description": "There are n rooms labeled from 0 to n - 1. Each room has a list of keys that can open other rooms. Return true if you can visit all rooms starting from room 0.",
        "solution": """def canVisitAllRooms(rooms):
    visited = {0}
    stack = [0]
    while stack:
        room = stack.pop()
        for key in rooms[room]:
            if key not in visited:
                visited.add(key)
                stack.append(key)
    return len(visited) == len(rooms)""",
        "solution_approach": "DFS with Stack",
        "examples": [
            {
                "input": "rooms = [[1],[2],[3],[]]",
                "output": "true",
                "explanation": "Can visit all rooms.",
            }
        ],
        "external_link": "https://leetcode.com/problems/keys-and-rooms/",
    },
    {
        "leetcode_id": 876,
        "title": "Middle of the Linked List",
        "difficulty": "Easy",
        "category": "Linked List",
        "description": "Given the head of a singly linked list, return the middle node of the list. If there are two middle nodes, return the second one.",
        "solution": """def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow""",
        "solution_approach": "Slow and Fast Pointers",
        "examples": [
            {
                "input": "head = [1,2,3,4,5]",
                "output": "[3,4,5]",
                "explanation": "Middle is node 3.",
            }
        ],
        "external_link": "https://leetcode.com/problems/middle-of-the-linked-list/",
    },
    {
        "leetcode_id": 897,
        "title": "Increasing Order Search Tree",
        "difficulty": "Easy",
        "category": "Tree",
        "description": "Given the root of a binary search tree, rearrange the tree in in-order such that the leftmost node becomes the new root, and every node has no left child.",
        "solution": """def increasingBST(root):
    dummy = TreeNode(-1)
    curr = dummy
    def inorder(node):
        if not node:
            return
        inorder(node.left)
        node.left = None
        curr.right = node
        curr = node
        inorder(node.right)
    inorder(root)
    return dummy.right""",
        "solution_approach": "In-order Traversal with Restructuring",
        "examples": [
            {
                "input": "root = [5,3,6,2,4,null,8,1,null,null,null,null,null,9,null,null,7]",
                "output": "[1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]",
                "explanation": "Tree restructured to right-only chain.",
            }
        ],
        "external_link": "https://leetcode.com/problems/increasing-order-search-tree/",
    },
    {
        "leetcode_id": 905,
        "title": "Sort Array By Parity",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given an integer array nums, move all the even integers at the beginning of the array followed by all the odd integers.",
        "solution": """def sortArrayByParity(nums):
    return [x for x in nums if x % 2 == 0] + [x for x in nums if x % 2 == 1]""",
        "solution_approach": "List Comprehension",
        "examples": [
            {
                "input": "nums = [3,1,2,4]",
                "output": "[2,4,3,1]",
                "explanation": "Even numbers first, then odd.",
            }
        ],
        "external_link": "https://leetcode.com/problems/sort-array-by-parity/",
    },
    {
        "leetcode_id": 994,
        "title": "Rotting Oranges",
        "difficulty": "Medium",
        "category": "BFS",
        "description": "Given a grid where each cell can be fresh (1), rotten (2), or empty (0), return the minimum number of minutes that must elapse until no cell is fresh. If impossible, return -1.",
        "solution": """def orangesRotting(grid):
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2:
                queue.append((i, j, 0))
            elif grid[i][j] == 1:
                fresh += 1
    minutes = 0
    while queue:
        i, j, minutes = queue.popleft()
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == 1:
                grid[ni][nj] = 2
                fresh -= 1
                queue.append((ni, nj, minutes + 1))
    return minutes if fresh == 0 else -1""",
        "solution_approach": "BFS Multi-source",
        "examples": [
            {
                "input": "grid = [[2,1,1],[1,1,0],[0,1,1]]",
                "output": "4",
                "explanation": "All oranges rot in 4 minutes.",
            }
        ],
        "external_link": "https://leetcode.com/problems/rotting-oranges/",
    },
    {
        "leetcode_id": 1025,
        "title": "Divisor Game",
        "difficulty": "Easy",
        "category": "Math",
        "description": "Alice and Bob take turns playing a game with a number N. Alice starts first. In each turn, a player chooses x (0 < x < N) such that N % x == 0 and subtracts x from N. Return true if Alice wins.",
        "solution": """def divisorGame(n):
    return n % 2 == 0""",
        "solution_approach": "Mathematical Insight",
        "examples": [
            {
                "input": "n = 2",
                "output": "true",
                "explanation": "Alice chooses 1, wins.",
            }
        ],
        "external_link": "https://leetcode.com/problems/divisor-game/",
    },
    {
        "leetcode_id": 1046,
        "title": "Last Stone Weight",
        "difficulty": "Easy",
        "category": "Heap",
        "description": "You are given an array of integers stones where stones[i] is the weight of the ith stone. On each turn, we choose the heaviest two stones and smash them together. Return the weight of the last remaining stone.",
        "solution": """import heapq

def lastStoneWeight(stones):
    stones = [-s for s in stones]
    heapq.heapify(stones)
    while len(stones) > 1:
        y = -heapq.heappop(stones)
        x = -heapq.heappop(stones)
        if y != x:
            heapq.heappush(stones, -(y - x))
    return -stones[0] if stones else 0""",
        "solution_approach": "Max Heap Simulation",
        "examples": [
            {
                "input": "stones = [2,7,4,1,8,1]",
                "output": "1",
                "explanation": "Last remaining stone weight is 1.",
            }
        ],
        "external_link": "https://leetcode.com/problems/last-stone-weight/",
    },
    {
        "leetcode_id": 1089,
        "title": "Duplicate Zeros",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given a fixed-length array arr of integers, duplicate each occurrence of zero, shifting the remaining elements to the right. Elements beyond the array length are discarded.",
        "solution": """def duplicateZeros(arr):
    i = 0
    length = len(arr)
    zeros = arr.count(0)
    for j in range(length - 1, -1, -1):
        if j + zeros < length:
            arr[j + zeros] = arr[j]
        if arr[j] == 0:
            zeros -= 1
            if j + zeros < length:
                arr[j + zeros] = 0""",
        "solution_approach": "Reverse Copy",
        "examples": [
            {
                "input": "arr = [1,0,2,3,0,4,5,6]",
                "output": "[1,0,0,2,3,0,0,4]",
                "explanation": "Zeros duplicated, extra elements discarded.",
            }
        ],
        "external_link": "https://leetcode.com/problems/duplicate-zeros/",
    },
    {
        "leetcode_id": 1108,
        "title": "Defanging an IP Address",
        "difficulty": "Easy",
        "category": "String",
        "description": "Given a valid (IPv4) IP address, return a defanged version of that IP address. A defanged IP address replaces every dot . with [.].",
        "solution": """def defangIPaddr(address):
    return address.replace(".", "[.]")""",
        "solution_approach": "String Replace",
        "examples": [
            {
                "input": "address = 1.1.1.1",
                "output": "1[.]1[.]1[.]1[.]1",
                "explanation": "Dots replaced with [.].",
            }
        ],
        "external_link": "https://leetcode.com/problems/defanging-an-ip-address/",
    },
    {
        "leetcode_id": 1122,
        "title": "Relative Sort Array",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given two arrays arr1 and arr2, sort arr1 such that the order of elements in arr1 should be the same as their order in arr2. Elements not in arr2 come at the end in sorted order.",
        "solution": """def relativeSortArray(arr1, arr2):
    from collections import Counter
    count = Counter(arr1)
    result = []
    for num in arr2:
        result.extend([num] * count[num])
        count[num] = 0
    for num in sorted(count):
        result.extend([num] * count[num])
    return result""",
        "solution_approach": "Counting Sort",
        "examples": [
            {
                "input": "arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,6,7]",
                "output": "[2,2,2,1,4,3,3,6,7,9,19]",
                "explanation": "arr1 sorted according to arr2 order.",
            }
        ],
        "external_link": "https://leetcode.com/problems/relative-sort-array/",
    },
    {
        "leetcode_id": 1207,
        "title": "Unique Number of Occurrences",
        "difficulty": "Easy",
        "category": "Hash Table",
        "description": "Given an array arr, return true if the number of distinct values in the array is equal to the number of distinct values in the occurrences array.",
        "solution": """def uniqueOccurrences(arr):
    from collections import Counter
    count = Counter(arr)
    return len(count.values()) == len(set(count.values()))""",
        "solution_approach": "Counter Comparison",
        "examples": [
            {
                "input": "arr = [1,2,2,1,1,3]",
                "output": "true",
                "explanation": "Occurrences: 1->3, 2->2, 3->1. All unique.",
            }
        ],
        "external_link": "https://leetcode.com/problems/unique-number-of-occurrences/",
    },
    {
        "leetcode_id": 1217,
        "title": "Move Chips",
        "difficulty": "Easy",
        "category": "Math",
        "description": "We have n chips, where the position of the ith chip is position[i]. We can move any chip to any position with the same parity. Return the minimum number of moves required.",
        "solution": """def minMovesToMoveChips(chips):
    even = sum(1 for c in chips if c % 2 == 0)
    odd = len(chips) - even
    return min(even, odd)""",
        "solution_approach": "Parity Counting",
        "examples": [
            {
                "input": "chips = [1,2,3]",
                "output": "2",
                "explanation": "Move 1 and 3 to position 2.",
            }
        ],
        "external_link": "https://leetcode.com/problems/minimum-moves-to-equal-array-elements/",
    },
    {
        "leetcode_id": 1221,
        "title": "Split a String in Balanced Strings",
        "difficulty": "Easy",
        "category": "Greedy",
        "description": "Given a balanced string s, split it into as many balanced substrings as possible. Return the maximum number of balanced substrings.",
        "solution": """def balancedStringSplit(s):
    balance = 0
    count = 0
    for char in s:
        if char == 'L':
            balance += 1
        else:
            balance -= 1
        if balance == 0:
            count += 1
    return count""",
        "solution_approach": "Balance Counter",
        "examples": [
            {
                "input": "s = RLRRLLRLRL",
                "output": "4",
                "explanation": "4 balanced substrings.",
            }
        ],
        "external_link": "https://leetcode.com/problems/split-a-string-in-balanced-strings/",
    },
    {
        "leetcode_id": 1281,
        "title": "Subtract the Product and Sum of Digits",
        "difficulty": "Easy",
        "category": "Math",
        "description": "Given an integer n, return the difference between the product of its digits and the sum of its digits.",
        "solution": """def subtractProductAndSum(n):
    product = 1
    sum_digits = 0
    while n:
        digit = n % 10
        product *= digit
        sum_digits += digit
        n //= 10
    return product - sum_digits""",
        "solution_approach": "Digit Extraction",
        "examples": [
            {
                "input": "n = 234",
                "output": "15",
                "explanation": "Product: 24, Sum: 9, Difference: 15.",
            }
        ],
        "external_link": "https://leetcode.com/problems/subtract-the-product-and-sum-of-digits-of-an-integer/",
    },
    {
        "leetcode_id": 1290,
        "title": "Convert Binary Linked List to Integer",
        "difficulty": "Easy",
        "category": "Linked List",
        "description": "Given the head of a singly linked list where each node represents a bit of the binary number, return its decimal value.",
        "solution": """def getDecimalValue(head):
    result = 0
    while head:
        result = result * 2 + head.val
        head = head.next
    return result""",
        "solution_approach": "Bit Accumulation",
        "examples": [
            {
                "input": "head = [1,0,1]",
                "output": "5",
                "explanation": "Binary 101 = 5.",
            }
        ],
        "external_link": "https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/",
    },
    {
        "leetcode_id": 1295,
        "title": "Find Numbers with Even Digits",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given an array of integers nums, return the count of numbers that have an even number of digits.",
        "solution": """def findNumbers(nums):
    count = 0
    for num in nums:
        if len(str(num)) % 2 == 0:
            count += 1
    return count""",
        "solution_approach": "Digit Count",
        "examples": [
            {
                "input": "nums = [12,345,2,6,7896]",
                "output": "2",
                "explanation": "12 and 7896 have even digits.",
            }
        ],
        "external_link": "https://leetcode.com/problems/find-numbers-with-even-number-of-digits/",
    },
    {
        "leetcode_id": 1313,
        "title": "Decompress Run-Length Encoded List",
        "difficulty": "Easy",
        "category": "Array",
        "description": "We are given a list of integers nums. The list contains pairs [freq, val]. Decompress it to create a new list.",
        "solution": """def decompressRLElist(nums):
    result = []
    for i in range(0, len(nums), 2):
        result.extend([nums[i+1]] * nums[i])
    return result""",
        "solution_approach": "List Extension",
        "examples": [
            {
                "input": "nums = [1,2,3,4]",
                "output": "[2,4,4,4]",
                "explanation": "2 repeated once, 4 repeated three times.",
            }
        ],
        "external_link": "https://leetcode.com/problems/decompress-run-length-encoded-list/",
    },
    {
        "leetcode_id": 1342,
        "title": "Steps to Reduce Number to Zero",
        "difficulty": "Easy",
        "category": "Math",
        "description": "Given a non-negative integer num, return the number of steps to reduce it to zero. If num is even, divide by 2. If odd, subtract 1.",
        "solution": """def numberOfSteps(num):
    steps = 0
    while num:
        if num % 2 == 0:
            num //= 2
        else:
            num -= 1
        steps += 1
    return steps""",
        "solution_approach": "Simulation",
        "examples": [
            {
                "input": "num = 14",
                "output": "6",
                "explanation": "14->7->6->3->2->1->0.",
            }
        ],
        "external_link": "https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero/",
    },
    {
        "leetcode_id": 1351,
        "title": "Count Negative Numbers in Sorted Matrix",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given a matrix grid sorted in non-increasing order, return the count of negative numbers.",
        "solution": """def countNegatives(grid):
    count = 0
    row = len(grid) - 1
    col = 0
    while row >= 0 and col < len(grid[0]):
        if grid[row][col] < 0:
            count += len(grid[0]) - col
            row -= 1
        else:
            col += 1
    return count""",
        "solution_approach": "Diagonal Traversal",
        "examples": [
            {
                "input": "grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]",
                "output": "8",
                "explanation": "8 negative numbers in the matrix.",
            }
        ],
        "external_link": "https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/",
    },
    {
        "leetcode_id": 1356,
        "title": "Sort by Number of 1 Bits",
        "difficulty": "Easy",
        "category": "Sorting",
        "description": "Given an integer array arr, sort arr by the number of 1-bits in the binary representation.",
        "solution": """def sortByBits(arr):
    return sorted(arr, key=lambda x: (bin(x).count('1'), x))""",
        "solution_approach": "Custom Sort",
        "examples": [
            {
                "input": "arr = [0,1,2,3,4,5,6,7,8]",
                "output": "[0,1,2,4,8,3,5,6,7]",
                "explanation": "Sorted by number of 1-bits.",
            }
        ],
        "external_link": "https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits/",
    },
    {
        "leetcode_id": 1365,
        "title": "Smaller Numbers Than Current",
        "difficulty": "Easy",
        "category": "Array",
        "description": "Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it.",
        "solution": """def smallerNumbersThanCurrent(nums):
    sorted_nums = sorted(nums)
    return [sorted_nums.index(x) for x in nums]""",
        "solution_approach": "Sorted Index Lookup",
        "examples": [
            {
                "input": "nums = [8,1,2,2,3]",
                "output": "[4,0,1,1,2]",
                "explanation": "8 has 4 smaller, 1 has 0, etc.",
            }
        ],
        "external_link": "https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/",
    },
    {
        "leetcode_id": 1370,
        "title": "Increase Decreasing String",
        "difficulty": "Easy",
        "category": "String",
        "description": "Given a string s, repeatedly remove one occurrence of the smallest character, then the largest character.",
        "solution": """def sortString(s):
    from collections import Counter
    count = Counter(s)
    result = []
    chars = sorted(count.keys())
    while count:
        for c in chars:
            if count[c] > 0:
                result.append(c)
                count[c] -= 1
        for c in reversed(chars):
            if count[c] > 0:
                result.append(c)
                count[c] -= 1
    return ''.join(result)""",
        "solution_approach": "Two-pass Construction",
        "examples": [
            {
                "input": "s = aaaabbbbcccc",
                "output": "abccbaabccba",
                "explanation": "String built by alternating smallest and largest.",
            }
        ],
        "external_link": "https://leetcode.com/problems/increase-decreasing-string/",
    },
]

SEED_ALGORITHMS = [
    {
        "name": "Binary Search",
        "category": "Search",
        "description": "A search algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search interval in half.",
        "implementation": """def binary_search(arr, target):
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
        "complexity": "O(log n)",
        "use_cases": [
            "Finding elements in sorted arrays",
            "Finding insertion points",
            "Finding floor/ceiling values",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Binary_search_algorithm",
    },
    {
        "name": "Quick Sort",
        "category": "Sorting",
        "description": "A divide-and-conquer algorithm that picks an element as a pivot and partitions the array around the picked pivot.",
        "implementation": """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",
        "complexity": "O(n log n) average, O(n²) worst case",
        "use_cases": [
            "General purpose sorting",
            "In-memory sorting",
            "External sorting with modifications",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Quicksort",
    },
    {
        "name": "Depth First Search",
        "category": "Graph",
        "description": "A graph traversal algorithm that explores as far as possible along each branch before backtracking.",
        "implementation": """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited""",
        "complexity": "O(V + E)",
        "use_cases": [
            "Path finding",
            "Cycle detection",
            "Topological sorting",
            "Solving mazes",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Depth-first_search",
    },
    {
        "name": "Breadth First Search",
        "category": "Graph",
        "description": "A graph traversal algorithm that explores all neighbors at the present depth before moving on to nodes at the next depth level.",
        "implementation": """from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited""",
        "complexity": "O(V + E)",
        "use_cases": [
            "Shortest path in unweighted graphs",
            "Level-order traversal",
            "Finding connected components",
            "Web crawlers",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Breadth-first_search",
    },
    {
        "name": "Dynamic Programming - Fibonacci",
        "category": "Dynamic Programming",
        "description": "A method for solving complex problems by breaking them down into simpler subproblems and storing their solutions.",
        "implementation": """def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]

# Bottom-up approach
def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b""",
        "complexity": "O(n)",
        "use_cases": [
            "Optimization problems",
            "Counting problems",
            "Sequence problems",
            "Game theory",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Dynamic_programming",
    },
    {
        "name": "Two Pointers",
        "category": "Two Pointers",
        "description": "A technique where two pointers traverse an array from different directions to solve problems efficiently.",
        "implementation": """def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []""",
        "complexity": "O(n)",
        "use_cases": [
            "Two sum in sorted array",
            "Trapping rain water",
            "Removing duplicates",
            "Finding triplets",
        ],
        "external_link": "https://www.geeksforgeeks.org/two-pointers-technique/",
    },
    {
        "name": "Sliding Window",
        "category": "Two Pointers",
        "description": "A technique that uses a window that slides through an array to efficiently solve problems involving subarrays.",
        "implementation": """def max_subarray_sum(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum""",
        "complexity": "O(n)",
        "use_cases": [
            "Maximum sum subarray",
            "Longest substring with conditions",
            "Average of all subarrays",
            "Minimum size subarray sum",
        ],
        "external_link": "https://www.geeksforgeeks.org/window-sliding-technique/",
    },
    {
        "name": "Merge Sort",
        "category": "Sorting",
        "description": "A divide-and-conquer algorithm that divides the input array into halves, recursively sorts them, and then merges the sorted halves.",
        "implementation": """def merge_sort(arr):
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
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result""",
        "complexity": "O(n log n)",
        "use_cases": [
            "External sorting",
            "Stable sorting required",
            "Linked list sorting",
            "Inversion counting",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Merge_sort",
    },
    {
        "name": "Heap Sort",
        "category": "Sorting",
        "description": "A comparison-based sorting algorithm that uses a binary heap data structure to sort elements.",
        "implementation": """def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)""",
        "complexity": "O(n log n)",
        "use_cases": [
            "Top K elements",
            "Priority queues",
            "Heap-based selection",
            "Median maintenance",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Heapsort",
    },
    {
        "name": "Hash Table",
        "category": "Data Structure",
        "description": "A data structure that maps keys to values and provides average O(1) time complexity for insertions, deletions, and lookups.",
        "implementation": """class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]
    def hash_function(self, key):
        return hash(key) % self.size
    def insert(self, key, value):
        idx = self.hash_function(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx][i] = (key, value)
                return
        self.table[idx].append((key, value))
    def get(self, key):
        idx = self.hash_function(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None
    def delete(self, key):
        idx = self.hash_function(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                del self.table[idx][i]
                return""",
        "complexity": "O(1) average, O(n) worst case",
        "use_cases": [
            "Caching",
            "Database indexing",
            "Counting frequencies",
            "Membership testing",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Hash_table",
    },
    {
        "name": "Backtracking",
        "category": "Search",
        "description": "A general algorithm for finding solutions by trying to build a solution incrementally and abandoning paths that fail.",
        "implementation": """def backtrack(path, choices, result):
    if is_solution(path):
        result.append(path[:])
        return
    for choice in choices:
        if is_valid(path, choice):
            path.append(choice)
            backtrack(path, choices, result)
            path.pop()

def solve_n_queens(n):
    result = []
    def is_valid(board, row, col):
        for i in range(row):
            if board[i] == col or board[i] - i == col - row or board[i] + i == col + row:
                return False
        return True
    def backtrack(row, cols, diag1, diag2, board):
        if row == n:
            result.append(["." * c + "Q" + "." * (n - c - 1) for c in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            backtrack(row + 1, cols | {col}, diag1 | {row - col}, diag2 | {row + col}, board + [col])
    backtrack(0, set(), set(), set(), [])
    return result""",
        "complexity": "O(N!) for N-Queens",
        "use_cases": [
            "N-Queens problem",
            "Sudoku solver",
            "Permutations/combinations",
            "Word search",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Backtracking",
    },
    {
        "name": "Union Find",
        "category": "Data Structure",
        "description": "A data structure that keeps track of a set of elements partitioned into disjoint subsets, supporting union and find operations.",
        "implementation": """class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
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
        return True
    def connected(self, x, y):
        return self.find(x) == self.find(y)""",
        "complexity": "O(α(n)) per operation (inverse Ackermann)",
        "use_cases": [
            "Connected components",
            "Network connectivity",
            "Kruskal's MST",
            "Image processing",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Disjoint-set_data_structure",
    },
    {
        "name": "Trie",
        "category": "Data Structure",
        "description": "A tree-like data structure used to store strings, where each node represents a character, enabling efficient prefix searches.",
        "implementation": """class TrieNode:
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
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True""",
        "complexity": "O(m) for operations, m = string length",
        "use_cases": [
            "Auto-complete",
            "Spell checking",
            "IP routing",
            "Word games",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Trie",
    },
    {
        "name": "Memoization",
        "category": "Optimization",
        "description": "An optimization technique that stores the results of expensive function calls and returns cached results when the same inputs occur again.",
        "implementation": """from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memoized(n):
    if n <= 1:
        return n
    return fib_memoized(n-1) + fib_memoized(n-2)

def edit_distance(s1, s2, memo={}):
    if (s1, s2) in memo:
        return memo[(s1, s2)]
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    if s1[0] == s2[0]:
        result = edit_distance(s1[1:], s2[1:], memo)
    else:
        result = 1 + min(
            edit_distance(s1[1:], s2, memo),
            edit_distance(s1, s2[1:], memo),
            edit_distance(s1[1:], s2[1:], memo)
        )
    memo[(s1, s2)] = result
    return result""",
        "complexity": "Depends on problem, reduces exponential to polynomial",
        "use_cases": [
            "Fibonacci sequence",
            "Edit distance",
            "Longest common subsequence",
            "Matrix chain multiplication",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Memoization",
    },
    {
        "name": "Greedy Algorithm",
        "category": "Optimization",
        "description": "An algorithmic paradigm that makes the locally optimal choice at each stage, hoping to find a global optimum.",
        "implementation": """def activity_selection(activities):
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    last_end = activities[0][1]
    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    return selected

def huffman_coding(frequencies):
    import heapq
    heap = [(freq, char) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(heap, (left[0] + right[0], (left, right)))
    return heap""",
        "complexity": "Varies by problem",
        "use_cases": [
            "Activity selection",
            "Huffman coding",
            "Fractional knapsack",
            "Dijkstra's algorithm",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Greedy_algorithm",
    },
    {
        "name": "Dijkstra's Algorithm",
        "category": "Graph",
        "description": "An algorithm for finding the shortest paths between nodes in a graph with non-negative edge weights.",
        "implementation": """import heapq

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        curr_dist, node = heapq.heappop(pq)
        if curr_dist > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return dist""",
        "complexity": "O((V + E) log V) with binary heap",
        "use_cases": [
            "Shortest path in weighted graphs",
            "Network routing",
            "GPS navigation",
            "Game pathfinding",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm",
    },
    {
        "name": "Bellman-Ford Algorithm",
        "category": "Graph",
        "description": "An algorithm for finding shortest paths from a source to all vertices in a weighted graph, handling negative edge weights.",
        "implementation": """def bellman_ford(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    V = len(graph)
    for _ in range(V - 1):
        for u in graph:
            for v, weight in graph[u]:
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
    for u in graph:
        for v, weight in graph[u]:
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                return None
    return dist""",
        "complexity": "O(V * E)",
        "use_cases": [
            "Negative weight edges",
            "Detecting negative cycles",
            "Arbitrage detection",
            "Shortest path with constraints",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm",
    },
    {
        "name": "Floyd-Warshall Algorithm",
        "category": "Graph",
        "description": "An algorithm for finding shortest paths between all pairs of vertices in a weighted graph.",
        "implementation": """def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u in graph:
        for v, w in graph[u]:
            dist[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist""",
        "complexity": "O(n³)",
        "use_cases": [
            "All-pairs shortest paths",
            "Transitive closure",
            "Finding negative cycles",
            "Matrix multiplication variants",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm",
    },
    {
        "name": "Kruskal's Algorithm",
        "category": "Graph",
        "description": "A greedy algorithm that finds a minimum spanning tree for a connected weighted graph.",
        "implementation": """import heapq

def kruskal(n, edges):
    parent = list(range(n))
    rank = [0] * n
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    edges = sorted(edges, key=lambda x: x[2])
    mst = []
    for u, v, w in edges:
        if union(u, v):
            mst.append((u, v, w))
    return mst""",
        "complexity": "O(E log E) or O(E log V)",
        "use_cases": [
            "Minimum spanning tree",
            "Network design",
            "Cluster analysis",
            "Image segmentation",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Kruskal%27s_algorithm",
    },
    {
        "name": "Prim's Algorithm",
        "category": "Graph",
        "description": "A greedy algorithm that finds a minimum spanning tree for a connected weighted graph by growing the tree from a starting vertex.",
        "implementation": """import heapq

def prim(n, graph, start=0):
    visited = [False] * n
    pq = [(0, start)]
    mst = []
    while pq:
        weight, node = heapq.heappop(pq)
        if visited[node]:
            continue
        visited[node] = True
        if weight > 0:
            mst.append((weight, node))
        for neighbor, w in graph[node]:
            if not visited[neighbor]:
                heapq.heappush(pq, (w, neighbor))
    return mst""",
        "complexity": "O(E log V)",
        "use_cases": [
            "Minimum spanning tree",
            "Network broadcasting",
            "Clustering",
            "Circuit design",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Prim%27s_algorithm",
    },
    {
        "name": "Bit Manipulation",
        "category": "Optimization",
        "description": "Techniques for manipulating individual bits in integer values, useful for optimization and specific algorithm implementations.",
        "implementation": """def count_bits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

def reverse_bits(n):
    result = 0
    for _ in range(32):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result

def swap_bits(n, i, j):
    if (n >> i) & 1 != (n >> j) & 1:
        n ^= (1 << i) | (1 << j)
    return n""",
        "complexity": "O(1) per operation",
        "use_cases": [
            "Efficient storage",
            "Bit masking",
            "Optimization problems",
            "Cryptography",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Bit_manipulation",
    },
    {
        "name": "Segment Tree",
        "category": "Data Structure",
        "description": "A tree data structure used for storing information about intervals, supporting range queries and updates efficiently.",
        "implementation": """class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
    def update(self, idx, value):
        pos = self.size + idx
        self.tree[pos] = value
        pos //= 2
        while pos:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos //= 2
    def query(self, l, r):
        l += self.size
        r += self.size
        result = 0
        while l <= r:
            if l % 2 == 1:
                result += self.tree[l]
                l += 1
            if r % 2 == 0:
                result += self.tree[r]
                r -= 1
            l //= 2
            r //= 2
        return result""",
        "complexity": "O(log n) for queries and updates",
        "use_cases": [
            "Range sum queries",
            "Range minimum/maximum",
            "Range updates",
            "Interval stabbing",
        ],
        "external_link": "https://en.wikipedia.org/wiki/Segment_tree",
    },
]


async def seed_database() -> None:
    """Seed the database with initial questions and algorithms."""
    await init_db()

    async with AsyncSessionLocal() as session:
        # Get existing question leetcode_ids
        result = await session.execute(select(QuestionDB.leetcode_id))
        existing_ids = {row[0] for row in result.fetchall()}

        # Add new questions that don't exist
        new_questions = [
            q for q in SEED_QUESTIONS if q["leetcode_id"] not in existing_ids
        ]
        if new_questions:
            logger.info(f"Seeding {len(new_questions)} new questions...")
            for q_data in new_questions:
                question = QuestionDB(**q_data)
                session.add(question)
            await session.commit()
            logger.info(
                f"Seeded {len(new_questions)} questions (total available: {len(SEED_QUESTIONS)})"
            )
        else:
            logger.info(f"All {len(SEED_QUESTIONS)} questions already exist")

        # Get existing algorithm names
        result = await session.execute(select(AlgorithmDB.name))
        existing_names = {row[0] for row in result.fetchall()}

        # Add new algorithms that don't exist
        new_algorithms = [a for a in SEED_ALGORITHMS if a["name"] not in existing_names]
        if new_algorithms:
            logger.info(f"Seeding {len(new_algorithms)} new algorithms...")
            for a_data in new_algorithms:
                algorithm = AlgorithmDB(**a_data)
                session.add(algorithm)
            await session.commit()
            logger.info(
                f"Seeded {len(new_algorithms)} algorithms (total available: {len(SEED_ALGORITHMS)})"
            )
        else:
            logger.info(f"All {len(SEED_ALGORITHMS)} algorithms already exist")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed_database())
