"""Pure manual mask rendering. Never opens or rewrites source/prediction files."""
import math,os,uuid
from pathlib import Path

def geometry(value,width,height):
    if not isinstance(value,dict) or set(value)!={'type','polygons'} or value['type']!='multi_polygon':raise ValueError('需要多多边形几何')
    polygons=value['polygons']
    if not isinstance(polygons,list) or len(polygons)>64:raise ValueError('轮廓数量不能超过64')
    result=[];total=0
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def intersects(a,b,c,d):
        def on(a,b,p):return abs(cross(a,b,p))<1e-9 and min(a[0],b[0])<=p[0]<=max(a[0],b[0]) and min(a[1],b[1])<=p[1]<=max(a[1],b[1])
        return cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0 or any([on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b)])
    for poly in polygons:
        if not isinstance(poly,list) or not 3<=len(poly)<=512:raise ValueError('每个轮廓须有3至512个点')
        total+=len(poly)
        if total>4096:raise ValueError('总点数不能超过4096')
        points=[]
        for p in poly:
            if not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p):raise ValueError('坐标须为有限数值')
            if not 0<=p[0]<=width or not 0<=p[1]<=height:raise ValueError('坐标超出原影像范围')
            points.append([float(p[0]),float(p[1])])
        if len(set(map(tuple,points)))!=len(points):raise ValueError('轮廓包含重复点')
        n=len(points)
        if abs(sum(points[i][0]*points[(i+1)%n][1]-points[(i+1)%n][0]*points[i][1] for i in range(n)))<1e-6:raise ValueError('轮廓不能退化为直线')
        for i in range(n):
            for j in range(i+1,n):
                if j==i+1 or (i==0 and j==n-1):continue
                if intersects(points[i],points[(i+1)%n],points[j],points[(j+1)%n]):raise ValueError('轮廓不能自交')
        result.append(points)
    return dict(type='multi_polygon',polygons=result)

def render_correction(root,request):
    from PIL import Image,ImageDraw
    import numpy as np
    import cv2
    if set(request)!={'correctionId','width','height','geometry'}:raise ValueError('修正参数无效')
    cid=request['correctionId']
    try:
        if str(uuid.UUID(cid))!=cid:raise ValueError()
    except (ValueError,TypeError,AttributeError):raise ValueError('修正标识须为UUID')
    width,height=request['width'],request['height']
    if type(width) is not int or type(height) is not int or min(width,height)<1 or width*height>16_777_216:raise ValueError('影像尺寸无效')
    validated=geometry(request['geometry'],width,height)
    mask=Image.new('L',(width,height),0);draw=ImageDraw.Draw(mask)
    for poly in validated['polygons']:draw.polygon([tuple(p) for p in poly],fill=255)
    binary=(np.array(mask)>0).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    objects=[];small_area=0
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i]);small_area+=area if area<16 else 0
        objects.append(dict(id=f'obj_{i:03d}',area_px=area,bbox=[x,y,x+w,y+h]))
    objects.sort(key=lambda v:v['area_px'],reverse=True)
    changed=int(binary.sum());ys,xs=np.where(binary)
    bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if changed else [0,0,0,0]
    bg=1-binary;cn,cl,cs,_=cv2.connectedComponentsWithStats(bg,connectivity=4)
    border=set(np.unique(np.concatenate([cl[0],cl[-1],cl[:,0],cl[:,-1]])).tolist())
    holes=[int(cs[i,cv2.CC_STAT_AREA]) for i in range(1,cn) if i not in border]
    # Same full-image denominator as the registered inference adapter.
    quality=dict(hole_count=len(holes),hole_ratio=sum(holes)/(width*height),small_fragment_count=sum(o['area_px']<16 for o in objects),small_fragment_area_ratio=small_area/max(changed,1))
    aid='correction-'+cid+'.png';target=Path(root)/aid
    if target.exists() or target.is_symlink():raise ValueError('修正资产已存在，不能覆盖')
    created=False
    try:
        with target.open('xb') as stream:
            created=True;mask.save(stream,format='PNG')
    except BaseException:
        # Remove only a file created by this call, never an existing prediction.
        if created and target.is_file():target.unlink()
        raise
    return dict(maskAssetId=aid,width=width,height=height,geometry=validated,changedPx=changed,
                changedRatio=changed/(width*height),objectCount=count-1,maskQuality=quality,bbox=bbox,objects=objects,
                qualityFlags=dict(has_holes=bool(holes),has_fragments=small_area>0),reviewReasonCodes=[],
                metricSource='manual_polygon;Pillow-inclusive-boundary;8-connected-components')
