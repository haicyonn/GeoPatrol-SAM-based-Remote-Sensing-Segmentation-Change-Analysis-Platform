package io.xunbian.backend;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

@Component
public class Storage {
    final Path root;
    public Storage(Settings settings) throws IOException {
        root=Path.of(settings.storageRoot).toAbsolutePath().normalize();Files.createDirectories(root);
    }
    public Path asset(String id){
        if(id==null||!id.matches("[A-Za-z0-9][A-Za-z0-9_.-]{0,190}")||id.equals(".")||id.equals(".."))
            throw new ApiError(400,"资产标识无效");
        Path file=root.resolve(id).normalize();
        try { if(!file.toRealPath().startsWith(root.toRealPath()))throw new ApiError(400,"资产越界"); }
        catch(NoSuchFileException e){return file;}
        catch(IOException e){throw new ApiError(400,"资产不可访问");}
        return file;
    }
    public record Stored(String id,int width,int height,long bytes,String mime){}
    public Stored save(MultipartFile file){
        if(file.isEmpty()||file.getSize()>50L*1024*1024)throw new ApiError(400,"影像为空或超过50MB");
        String name=Optional.ofNullable(file.getOriginalFilename()).orElse("");
        String ext=name.contains(".")?name.substring(name.lastIndexOf('.')).toLowerCase(Locale.ROOT):"";
        if(!Set.of(".png",".jpg",".jpeg",".tif",".tiff",".bmp").contains(ext))throw new ApiError(400,"影像格式不支持");
        String id=UUID.randomUUID()+ext;Path path=asset(id);
        try {
            file.transferTo(path);
            try(var input=ImageIO.createImageInputStream(path.toFile())){
                var readers=ImageIO.getImageReaders(input);
                if(!readers.hasNext())throw new ApiError(400,"影像无法解码");
                var reader=readers.next();
                try{
                    reader.setInput(input);int w=reader.getWidth(0),h=reader.getHeight(0);
                    if(w<1||h<1||(long)w*h>16_777_216)throw new ApiError(400,"单张影像不得超过16777216像素");
                    BufferedImage decoded=reader.read(0);
                    if(decoded==null)throw new ApiError(400,"影像无法完整解码");
                    return new Stored(id,w,h,Files.size(path),"image/"+(ext.equals(".jpg")?"jpeg":ext.substring(1)));
                }finally{reader.dispose();}
            }
        }catch(Exception e){
            try{Files.deleteIfExists(path);}catch(IOException ignored){}
            if(e instanceof ApiError a)throw a;
            throw new ApiError(400,"影像读取失败");
        }
    }
    public void removeUncommitted(Collection<String> ids){
        for(String id:ids)try{Files.deleteIfExists(asset(id));}catch(IOException ignored){}
    }
}
