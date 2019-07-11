# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:55:35 2019

@author: Abhi
"""

import unittest
import seg

class TestSeg(unittest.TestCase):
    
    def test_met(self):
        
        txt = {}
        act = seg.met(txt)
        exp = {}
        self.assertEqual(act, exp)
        
        txt = {"b++a"}
        act = seg.met(txt)
        exp = {"a"}
        self.assertEqual(act, exp)
        
        txt = {"b++a","a++b"}
        act = seg.met(txt)
        exp = {"a","b"}
        self.assertEqual(act, exp)
    
    def test_seg(self):
        
        act = seg.seg(r"emptyText.txt")
        exp = {}
        self.assertEqual(act, exp)
        
        act = seg.seg(r"testFile1.txt")
        exp = {"hello"}
        self.assertEqual(act, exp)
        
        act = seg.seg(r"testFile2.txt")
        exp = {"hello", "world"}
        self.assertEqual(act, exp)
    
    def test_token(self):
        
        act = seg.seg(r"emptyText.txt")
        exp = {}
        self.assertEqual(act, exp)
        
        act = seg.seg(r"testFile1.txt")
        exp = {"hello"}
        self.assertEqual(act, exp)
        
        act = seg.seg(r"testFile2.txt")
        exp = {"hello", "world"}
        self.assertEqual(act, exp)
        
        act = seg.seg(r"testFile3.txt")
        exp = {"this", "is", "a", "sentence","."}
        self.assertEqual(act, exp)
    
    def test_lem(self):
        
        act = seg.seg(r"emptyText.txt")
        exp = {}
        self.assertEqual(act, exp)
        
        act = seg.seg(r"testFile1.txt")
        exp = {"hello"}
        self.assertEqual(act, exp)
        
        act = seg.seg(r"testFile4(lem).txt")
        exp = {"write", "write"}
        self.assertEqual(act, exp)
        
        