import unittest
import Density

class DensityTest(unittest.TestCase):
    testFileName ='testDensity.dat'
    testObject = None
    def test_initializing(self):
        anObject = Density.Density('testDensity.dat')
        self.assertEqual(50,self.testObject.getSize())
        # provide a density file
        self.assertAlmostEqual(0.020408163,self.testObject.epsilon, 8)
        # test some values

    def test_access(self):

        # create density Object
        self.assertEqual(4.969467, self.testObject.getDensityAtXY(0,0))
        self.assertAlmostEqual(5.556544, self.testObject.getDensityAtXY(0.020408, 0.12249),6)
        self.assertEqual(4.969467, self.testObject.getDensityAtXY(0.0,0.0))
        self.assertEqual(5.679870, self.testObject.getDensityAtXY(0.020408, 0.653061))
        self.assertEqual(self.testObject.getDensityAtXY(0.5,0.5),self.testObject.getDensityAtXY(1.5,2.5))
        self.assertEqual(self.testObject.getDensityAtXY(0.1025,0.5123),self.testObject.getDensityAtXY(1.1025,2.5123))
        self.assertEqual(self.testObject.getDensityAtXY(0.1025-1.0,0.5123),self.testObject.getDensityAtXY(1.1025,2.5123))
        self.assertEqual(self.testObject.getDensityAtXY(0.1025-1.0,0.5123+1),self.testObject.getDensityAtXY(1.1025,2.5123))
        # access some non-grid values
        self.assertEqual(5.679870, self.testObject.getDensityAtXY(0.021408, 0.66))
    def testCoordinatenUmrechnung(self):
        x = 0.020408
        y = 0.020408
        newx,newy = self.testObject.xyFromCoordtoArray(x,y)
        self.assertEqual(newx,1)
        self.assertEqual(newy,1)
        x = 0.979592
        y = 0.979592
        newx,newy = self.testObject.xyFromCoordtoArray(x,y)
        self.assertEqual(newx,48)
        self.assertEqual(newy,48)
        newx,newy = self.testObject.xyFromCoordtoArray(1.0,1.0)
        self.assertEqual(newx,49)
        self.assertEqual(newy,49)
        newx,newy = self.testObject.xyFromCoordtoArray(0.0,0.0)
        self.assertEqual(newx,0)
        self.assertEqual(newy,0)


    def testChargeinArea(self):
        self.assertAlmostEqual(5.679870*0.000416491, self.testObject.getChargeInArea(0.021408, 0.66),7)
    def testFullCharge(self):
        self.assertAlmostEqual(6.0,self.testObject.getChargeInUnitCell(),7)
    def test515(self):
        newTestObject = Density.Density("density5-15.dat")
        self.assertEqual(100,newTestObject.size)
        self.assertEqual(1/99.0, newTestObject.epsilon)
        self.assertAlmostEqual(5.0, newTestObject.getChargeInUnitCell(),3)
# setup here
        
    def setUp(self):
        self.testObject= Density.Density(self.testFileName)
        
if __name__ == '__main__':
    unittest.main()
