package io.xunbian.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.SecureRandom;
import java.util.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.core.*;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.*;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

@Service
public class AccountService implements UserDetailsService {
    public record Actor(String id,String username,String displayName,String role,long version) implements java.io.Serializable {
        public boolean admin(){return "ADMIN".equals(role);}
    }
    public record Principal(Actor actor,String hash,boolean active) implements UserDetails {
        public Collection<? extends GrantedAuthority> getAuthorities(){return List.of(new SimpleGrantedAuthority("ROLE_"+actor.role()));}
        public String getPassword(){return hash;} public String getUsername(){return actor.username();}
        public boolean isEnabled(){return active;}
    }
    final JdbcTemplate db; final TransactionTemplate tx;
    final BCryptPasswordEncoder passwords=new BCryptPasswordEncoder(12);
    public AccountService(JdbcTemplate db,TransactionTemplate tx,SchemaMigration migration,ObjectMapper json,
        @Value("${xunbian.bootstrap-file:${user.home}/.config/xunbian-backend/initial-admin.json}") String bootstrapFile){
        this.db=db;this.tx=tx;
        tx.executeWithoutResult(t->{
            db.queryForObject("SELECT id FROM app_bootstrap WHERE id=1 FOR UPDATE",Integer.class);
            if(db.queryForObject("SELECT COUNT(*) FROM app_users",Integer.class)==0){
                byte[] bytes=new byte[24];new SecureRandom().nextBytes(bytes);
                String password=Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
                insert("admin","课题组管理员",password,"ADMIN");
                try{
                    Path path=Path.of(bootstrapFile);Files.createDirectories(path.getParent());
                    Path tmp=Files.createTempFile(path.getParent(),".initial-admin-",".tmp");
                    try{Files.setPosixFilePermissions(tmp,PosixFilePermissions.fromString("rw-------"));}catch(UnsupportedOperationException ignored){}
                    Files.writeString(tmp,json.writeValueAsString(Map.of("username","admin","password",password,
                        "notice","首次登录后请修改密码；此文件仅记录初始密码。")),StandardCharsets.UTF_8);
                    Files.move(tmp,path,StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);
                }catch(Exception e){throw new IllegalStateException("无法保存私有管理员初始凭据",e);}
            }
            String admin=db.queryForObject("SELECT id FROM app_users WHERE role='ADMIN' ORDER BY created_ms LIMIT 1",String.class);
            db.update("UPDATE batches SET owner_id=? WHERE owner_id IS NULL",admin);
        });
    }
    Principal from(Map<String,Object> r){return new Principal(new Actor(r.get("id").toString(),r.get("username").toString(),r.get("display_name").toString(),r.get("role").toString(),((Number)r.get("auth_version")).longValue()),r.get("password_hash").toString(),Boolean.TRUE.equals(r.get("enabled"))||"1".equals(String.valueOf(r.get("enabled"))));}
    public Principal loadUserByUsername(String username){
        var rows=db.queryForList("SELECT * FROM app_users WHERE username=?",username);
        if(rows.isEmpty())throw new UsernameNotFoundException("账号或密码错误");return from(rows.getFirst());
    }
    public Actor current(){
        Authentication a=SecurityContextHolder.getContext().getAuthentication();
        if(a==null||!a.isAuthenticated()||!(a.getPrincipal() instanceof Principal p))throw new ApiError(401,"请先登录");
        Principal latest=loadUserByUsername(p.actor().username());
        if(!latest.active()||latest.actor().version()!=p.actor().version())throw new ApiError(401,"账号状态已变化，请重新登录");
        return p.actor();
    }
    public Actor requireAdmin(){Actor a=current();if(!a.admin())throw new ApiError(403,"仅管理员可操作");return a;}
    public static void validPassword(String password){
        if(password==null||password.length()<10||password.getBytes(StandardCharsets.UTF_8).length>72)throw new ApiError(400,"密码需至少10个字符，UTF-8长度不超过72字节");
    }
    String insert(String username,String display,String password,String role){
        String id=UUID.randomUUID().toString();db.update("INSERT INTO app_users(id,username,display_name,password_hash,role,created_ms) VALUES(?,?,?,?,?,?)",id,username,display,passwords.encode(password),role,System.currentTimeMillis());return id;
    }
    public Map<String,Object> create(String username,String display,String password,String role){
        requireAdmin();validPassword(password);
        if(username==null||!username.matches("[a-zA-Z0-9][a-zA-Z0-9_.-]{2,31}")||display==null||display.isBlank()||display.length()>60||!Set.of("ADMIN","MEMBER").contains(role))throw new ApiError(400,"账号、姓名或角色无效");
        try{String id=insert(username,display,password,role);return Map.of("id",id,"username",username,"displayName",display,"role",role,"enabled",true);}
        catch(org.springframework.dao.DuplicateKeyException e){throw new ApiError(409,"账号已存在");}
    }
    public List<Map<String,Object>> list(){requireAdmin();return db.queryForList("SELECT id,username,display_name AS displayName,role,enabled FROM app_users ORDER BY created_ms");}
    public void enabled(String id,boolean enabled){
        Actor a=requireAdmin();if(a.id().equals(id)&&!enabled)throw new ApiError(409,"不能禁用当前管理员本人");
        tx.executeWithoutResult(t->{
            db.queryForObject("SELECT id FROM app_bootstrap WHERE id=1 FOR UPDATE",Integer.class);
            var rows=db.queryForList("SELECT role FROM app_users WHERE id=? FOR UPDATE",id);
            if(rows.isEmpty())throw new ApiError(404,"账号不存在");
            if(!enabled&&"ADMIN".equals(rows.getFirst().get("role"))&&db.queryForObject("SELECT COUNT(*) FROM app_users WHERE role='ADMIN' AND enabled=TRUE",Integer.class)<=1)throw new ApiError(409,"至少保留一个可用管理员");
            db.update("UPDATE app_users SET enabled=?,auth_version=auth_version+1 WHERE id=?",enabled,id);
        });
    }
    public void changePassword(String old,String password){
        Actor a=current();validPassword(password);Principal p=loadUserByUsername(a.username());
        if(old==null||!passwords.matches(old,p.hash()))throw new ApiError(403,"当前密码不正确");
        db.update("UPDATE app_users SET password_hash=?,auth_version=auth_version+1 WHERE id=?",passwords.encode(password),a.id());
    }
    public void resetPassword(String id,String password){
        Actor a=requireAdmin();validPassword(password);
        if(a.id().equals(id))throw new ApiError(409,"请通过个人密码修改功能更新自己的密码");
        if(db.update("UPDATE app_users SET password_hash=?,auth_version=auth_version+1 WHERE id=?",passwords.encode(password),id)!=1)throw new ApiError(404,"账号不存在");
    }
}
