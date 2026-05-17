-- Super Agent MySQL 初始化脚本
-- 会话记忆表

CREATE DATABASE IF NOT EXISTS super;
USE super;

-- 对话会话表
CREATE TABLE IF NOT EXISTS conversation (
    id VARCHAR(64) PRIMARY KEY,
    user_id INT NOT NULL DEFAULT 0,
    title VARCHAR(255) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    summary TEXT,
    INDEX idx_user_id (user_id),
    INDEX idx_updated_at (updated_at),
    INDEX idx_user_updated (user_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 对话消息表
CREATE TABLE IF NOT EXISTS message (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id VARCHAR(64) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (conversation_id) REFERENCES conversation(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 文档索引记录表
CREATE TABLE IF NOT EXISTS document_index (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL,
    document_name VARCHAR(255),
    file_path VARCHAR(512),
    file_size BIGINT,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_document_id (document_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;