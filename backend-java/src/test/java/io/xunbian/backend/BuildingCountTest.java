package io.xunbian.backend;

import java.awt.image.BufferedImage;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BuildingCountTest {
    static BufferedImage mask(){return new BufferedImage(40,30,BufferedImage.TYPE_BYTE_GRAY);}
    static void box(BufferedImage image,int x,int y,int w,int h){for(int yy=y;yy<y+h;yy++)for(int xx=x;xx<x+w;xx++)image.getRaster().setSample(xx,yy,0,255);}
    @Test void emptyMaskIsZero(){var r=BuildingCountService.measure(mask());assertEquals(0,r.get("estimatedBuildingCount"));assertEquals(0,r.get("filteredFragmentCount"));}
    @Test void filtersOnlySmallComponentsAndRetainsBoundaryEvidence(){
        var image=mask();box(image,2,2,5,5);box(image,12,3,3,3);box(image,24,0,2,5);
        var r=BuildingCountService.measure(image);
        assertEquals(3,r.get("rawComponentCount"));assertEquals(2,r.get("estimatedBuildingCount"));
        assertEquals(1,r.get("filteredFragmentCount"));assertEquals(1,r.get("boundaryBuildingCount"));
        assertEquals(10,r.get("countMinAreaPixels"));
        var candidates=(List<Map<String,Object>>)r.get("buildingCandidates");
        assertEquals(2,candidates.size());assertEquals(List.of(2,2,7,7),candidates.get(1).get("bbox"));
        assertEquals(255,image.getRaster().getSample(12,3,0),"counting must not remove pixels from the original mask");
    }
    @Test void diagonalConnectionUsesEightNeighboursAndDoesNotSplitTouchingRoofs(){
        var image=mask();box(image,2,2,3,2);box(image,5,4,3,2);
        var r=BuildingCountService.measure(image);assertEquals(1,r.get("rawComponentCount"));assertEquals(1,r.get("estimatedBuildingCount"));
    }
}
