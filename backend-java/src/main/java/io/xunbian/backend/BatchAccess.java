package io.xunbian.backend;

import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

@Service
public class BatchAccess {
    final AccountService accounts;final JdbcTemplate db;
    public BatchAccess(AccountService accounts,JdbcTemplate db){this.accounts=accounts;this.db=db;}
    Map<String,Object> batch(String id){var rows=db.queryForList("SELECT owner_id,source_type FROM batches WHERE id=?",id);if(rows.isEmpty())throw new ApiError(404,"批次不存在");return rows.getFirst();}
    public String permission(String id){
        var a=accounts.current();var b=batch(id);
        if(a.admin()||a.id().equals(b.get("owner_id")))return "OWNER";
        var rows=db.queryForList("SELECT permission FROM batch_members WHERE batch_id=? AND user_id=?",id,a.id());
        return rows.isEmpty()?"NONE":rows.getFirst().get("permission").toString();
    }
    public void requireRead(String id){if("NONE".equals(permission(id)))throw new ApiError(403,"无权访问该批次");}
    public void requireEdit(String id){if(!Set.of("OWNER","EDIT").contains(permission(id)))throw new ApiError(403,"无权修改该批次");}
    public void requireOwner(String id){if(!"OWNER".equals(permission(id)))throw new ApiError(403,"仅批次创建者或管理员可管理协作者");}
    public Object members(String id){
        requireOwner(id);
        return Map.of("members",db.queryForList("SELECT u.id,u.username,u.display_name AS displayName,m.permission FROM batch_members m JOIN app_users u ON u.id=m.user_id WHERE m.batch_id=?",id),
            "availableUsers",db.queryForList("SELECT id,username,display_name AS displayName FROM app_users WHERE enabled=TRUE ORDER BY username"));
    }
    public void grant(String id,String user,String permission){
        requireOwner(id);if(!Set.of("VIEW","EDIT","NONE").contains(permission))throw new ApiError(400,"权限仅支持VIEW、EDIT或NONE");
        if(db.queryForObject("SELECT COUNT(*) FROM app_users WHERE id=? AND enabled=TRUE",Integer.class,user)==0)throw new ApiError(404,"成员不存在或已禁用");
        if("NONE".equals(permission))db.update("DELETE FROM batch_members WHERE batch_id=? AND user_id=?",id,user);
        else db.update("INSERT INTO batch_members VALUES(?,?,?) ON DUPLICATE KEY UPDATE permission=VALUES(permission)",id,user,permission);
    }
}
