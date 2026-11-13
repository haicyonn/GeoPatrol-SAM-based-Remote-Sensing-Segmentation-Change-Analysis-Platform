import sys,tempfile,unittest,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from PIL import Image
from manual_corrections import render_correction

class ManualCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def request(self,polygons):
        return dict(correctionId=str(uuid.uuid4()),width=32,height=32,geometry=dict(type='multi_polygon',polygons=polygons))
    def test_square_uses_original_pixel_coordinates_and_new_mask(self):
        r=render_correction(self.root,self.request([[[2,2],[6,2],[6,6],[2,6]]]))
        self.assertEqual(r['changedPx'],25);self.assertEqual(r['bbox'],[2,2,7,7]);self.assertEqual(r['objectCount'],1)
        with Image.open(self.root/r['maskAssetId']) as image:self.assertEqual(image.size,(32,32))
    def test_union_and_empty_do_not_create_duplicate_area_or_fake_object(self):
        poly=[[2,2],[6,2],[6,6],[2,6]]
        self.assertEqual(render_correction(self.root,self.request([poly,poly]))['changedPx'],25)
        r=render_correction(self.root,self.request([]));self.assertEqual(r['changedPx'],0);self.assertEqual(r['objectCount'],0)
    def test_bad_geometry_is_rejected_without_files(self):
        for polygon in ([[0,0],[33,0],[0,3]],[[0,0],[1,1],[2,2]],[[1,1],[8,8],[1,8],[8,1]],[[0,0],[float('nan'),1],[2,2]]):
            with self.subTest(polygon=polygon),self.assertRaises(ValueError):render_correction(self.root,self.request([polygon]))
        self.assertEqual(list(self.root.iterdir()),[])
    def test_existing_asset_and_path_escape_are_never_overwritten(self):
        req=self.request([]);render_correction(self.root,req)
        with self.assertRaises(ValueError):render_correction(self.root,req)
        req['correctionId']='../bad'
        with self.assertRaises(ValueError):render_correction(self.root,req)

if __name__=='__main__':unittest.main()
