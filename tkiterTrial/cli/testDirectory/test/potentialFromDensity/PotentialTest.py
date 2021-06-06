import unittest
import Potential
import numpy as np

class PotentialTest(unittest.TestCase):

    def test_init(self):
        anObject = Potential.Potential('testDensity.dat')
        self.assertTrue(anObject.density)
        self.assertAlmostEqual(9.7081,anObject.cellSizeLo,2)
        self.assertEqual(5, anObject.Ne)
        self.assertEqual(15, anObject.Nm)
        self.assertAlmostEqual(0.33333, anObject.fillingFactor,3)
        print ("Done")
    def testDistance(self):
        testObject =   Potential.Potential('testDensity10_by_10.dat')
        self.assertAlmostEqual(testObject.getDistance(0.0,0.0,0.1,0.3),3.06998, 4)
        self.assertAlmostEqual(testObject.getDistance(0.1,0.3,0.0,0.0),3.06998, 4)
    def testGetPotContribution(self):
        anObject = Potential.Potential('testDensity.dat')
        pot = anObject.getPotentialContribution(0.2,0.1 ,0.591837, 0.632653 )
        print (pot)
        self.assertAlmostEqual(0.18098962, anObject.getPotentialContribution(0.2,0.1, 0.020408, 0.12249),6)


    def testgetPotentialCoarse(self):
        testObject =   Potential.Potential('testDensity10_by_10.dat')
        self.assertAlmostEqual(0.05075644,testObject.density.getChargeInArea(0.0,0.0),4)
        self.assertAlmostEqual(0.016533,testObject.getPotentialContribution(0.1,0.3,0.0,0.0),4)
        self.assertAlmostEqual(0.0165654458,testObject.getPotentialContribution(0.1,0.3,0.2,0.6),4)
#        self.assertAlmostEqual(0.016533,testObject.getPotentialContribution(0.1,0.3,0.1,0.3),4)
        self.assertAlmostEqual(1.564118851, testObject.getPotentialatXY(0.1,0.3),3)
        
    def testgetPotentialPBCCoarse(self):
        testObject =   Potential.Potential('testDensity10_by_10.dat')
        pot =  testObject.getPotentialatXY(0.1,0.3)
        self.assertAlmostEqual(1.564118851, testObject.getPotentialatXY(0.1,0.3),3)
        potPBC = testObject.getPotentialAtXYPBC(0.1,0.3)
        
        self.assertTrue(np.abs(potPBC) > np.abs(testObject.getPotentialatXY(0.1,0.3)))
        apbc = testObject.getPotentialAtXYPBC(0.6,0.6)
        print ("Difference")
        print (apbc, pot)
