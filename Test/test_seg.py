# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:55:35 2019

@author: Abhi
"""

import unittest
import seg

class TestSeg(unittest.TestCase):
    
    def test_load_new_file(self):
        
        """ act = seg.load_new_file(r"emptyText.txt")
        exp = ['']
        self.assertEqual(act, exp)
        
        act = seg.load_new_file(r"testFile1.txt")
        exp = ["hello"]
        self.assertEqual(act, exp)
        
        act = seg.load_new_file(r"testFile2.txt")
        exp = ["hello", "world"]
        self.assertEqual(act, exp) """

        act = seg.slice(r"emptyText.txt")
        exp = ['']
        self.assertEqual(act, exp)
        
        act = seg.slice(r"testFile1.txt")
        exp = ["hello"]
        self.assertEqual(act, exp)
        
        act = seg.slice(r"testFile2.txt")
        exp = ["hello", "world"]
        self.assertEqual(act, exp)

        act = seg.slice(r"slice_test.txt")
        exp = [""" 
            Our Applause customers rely on uTesters around the world to be able to provide real-world feedback on an increasingly wide array of devices. Today we are spotlighting a type of device that has been gaining popularity amongst the devices needing to be tested on uTest: Virtual reality systems.
            """,
            """
            Tokyo 2020: Meet the Olympic and Paralympic robots
            What are your favorite Olympic sports? Are you excited for the 2020 Tokyo Olympics?
            Tokyo will use a Robots as Olympic mascots to greet people when they get to venues and to give virtual access to people who can't attend the games.
            """ ]
        self.assertEqual(act, exp)

# =============================================================================
#     def test_load_pickled(self):
#         
#         act = seg.load_pickled(r"emptyText.txt")
#         exp = ['']
#         self.assertEqual(act, exp)
#         
#         act = seg.load_pickled(r"testFile1.txt")
#         exp = ["hello"]
#         self.assertEqual(act, exp)
#         
#         act = seg.load_pickled(r"testFile2.txt")
#         exp = ["hello", "world"]
#         self.assertEqual(act, exp)
# =============================================================================
        
"""     def test_load_file(self):
        
        act = seg.load_file(r"emptyText.txt")
        exp = ['']
        self.assertEqual(act, exp)
        
        act = seg.load_file(r"testFile1.txt")
        exp = ["hello"]
        self.assertEqual(act, exp)
        
        act = seg.load_file(r"testFile2.txt")
        exp = ["hello", "world"]
        self.assertEqual(act, exp)
    
    def test_token(self):
        
        act = seg.token(r"emptyText.txt")
        exp = ['']
        self.assertEqual(act, exp)
        
        act = seg.token(r"testFile1.txt")
        exp = ["hello"]
        self.assertEqual(act, exp)
        
        act = seg.token(r"testFile2.txt")
        exp = ["hello", "world"]
        self.assertEqual(act, exp)
        
        act = seg.token(r"testFile3.txt")
        exp = ["this", "is", "a", "sentence","."]
        self.assertEqual(act, exp)
    
    def test_lem(self):
        
        act = seg.lem(r"emptyText.txt")
        exp = ['']
        self.assertEqual(act, exp)
        
        act = seg.lem(r"testFile1.txt")
        exp = ["hello"]
        self.assertEqual(act, exp)
        
        act = seg.lem(r"testFile4(lem).txt")
        exp = ["write", "write"]
        self.assertEqual(act, exp) """
        
if __name__ == '__main__':
    unittest.main()