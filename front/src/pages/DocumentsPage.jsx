import { useState, useEffect } from 'react'
import { Upload, FileText, Trash2, RefreshCw, CheckCircle, XCircle, Clock, Layers, Zap, Brain, Cpu, Loader2 } from 'lucide-react'
import { documentApi } from '../api/services'
import { useAuth } from '../context/AuthContext'
import './DocumentsPage.css'

const UPLOAD_STATUS = { PENDING: 'pending', UPLOADING: 'uploading', SUCCESS: 'success', FAILED: 'failed' }

const STRATEGY_OPTIONS = [
  {
    value: 'structural,recursive',
    label: '结构+递归',
    icon: Layers,
    desc: '按文档标题切分，超限递归裁切。适合大部分文档',
    tag: '推荐'
  },
  {
    value: 'structural,recursive,semantic',
    label: '结构+递归+语义',
    icon: Zap,
    desc: '在结构切块基础上，自动检测主题边界精修。适合复杂长文档',
    tag: '精修'
  },
  {
    value: 'structural,recursive,semantic,llm',
    label: '全策略流水线',
    icon: Brain,
    desc: '包含 LLM 智能切块，处理低质量/排版混乱的文档。需额外 API 调用',
    tag: '增强'
  },
  {
    value: 'structural',
    label: '仅结构切块',
    icon: Cpu,
    desc: '纯粹按文档标题/段落切分，不做额外处理',
    tag: '轻量'
  }
]

function DocumentsPage() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [selectedStrategy, setSelectedStrategy] = useState('structural,recursive')
  const [uploadTasks, setUploadTasks] = useState([])

  const { user } = useAuth()
  const userId = user?.user_id || parseInt(localStorage.getItem('user_id')) || 0

  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    setLoading(true)
    try {
      const res = await documentApi.list(userId)
      if (res.success) {
        setDocuments(res.documents || [])
      }
    } catch (error) {
      console.error('Failed to load documents:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return

    const fileList = Array.from(files)
    const initialTasks = fileList.map((file, i) => ({
      id: Date.now() + i,
      name: file.name,
      status: UPLOAD_STATUS.PENDING,
    }))

    setUploadTasks(prev => [...prev, ...initialTasks])

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i]
      const taskId = initialTasks[i].id

      // 更新为上传中
      setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: UPLOAD_STATUS.UPLOADING } : t))

      try {
        const res = await documentApi.upload(file, userId, null, selectedStrategy)
        if (res.success) {
          setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: UPLOAD_STATUS.SUCCESS } : t))
        } else {
          const errMsg = res.document?.status === 'parse_failed' ? '解析失败' : '处理失败'
          setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: UPLOAD_STATUS.FAILED, error: errMsg } : t))
        }
      } catch (error) {
        setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: UPLOAD_STATUS.FAILED, error: '网络错误' } : t))
      }
    }

    await loadDocuments()
  }

  const dismissTask = (taskId) => {
    setUploadTasks(prev => prev.filter(t => t.id !== taskId))
  }

  const clearCompletedTasks = () => {
    setUploadTasks(prev => prev.filter(t => t.status === UPLOAD_STATUS.UPLOADING || t.status === UPLOAD_STATUS.PENDING))
  }

  const handleFileSelect = (e) => {
    handleUpload(e.target.files)
    e.target.value = ''
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleUpload(e.dataTransfer.files)
  }

  const handleDelete = async (docId) => {
    if (!confirm('确定要删除这个文档吗？')) return
    try {
      const res = await documentApi.delete(docId)
      if (res.success) {
        setDocuments(prev => prev.filter(d => d.id !== docId))
      }
    } catch (error) {
      alert('删除失败')
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'indexed': return <CheckCircle size={16} className="status-icon success" />
      case 'keyword_only': return <CheckCircle size={16} className="status-icon warning" />
      case 'error': return <XCircle size={16} className="status-icon error" />
      default: return <Clock size={16} className="status-icon pending" />
    }
  }

  const getStatusText = (status) => {
    switch (status) {
      case 'indexed': return '已索引'
      case 'keyword_only': return '仅关键词'
      case 'error': return '失败'
      default: return '处理中'
    }
  }

  const getStrategyLabel = (strategy) => {
    if (!strategy) return '默认'
    const names = strategy.split(',').map(s => {
      switch (s.trim()) {
        case 'structural': return '结构'
        case 'recursive': return '递归'
        case 'semantic': return '语义'
        case 'llm': return 'LLM'
        default: return s
      }
    })
    return names.join('+')
  }

  return (
    <div className="documents-page">
      <header className="page-header glass">
        <div className="header-left">
          <h1>文档管理</h1>
          <span className="header-subtitle">上传和管理知识库文档</span>
        </div>
        <div className="header-actions">
          <button className="refresh-btn" onClick={loadDocuments} disabled={loading}>
            <RefreshCw size={18} className={loading ? 'spin' : ''} />
          </button>
        </div>
      </header>

      <div className="content-area">
        {/* 切块策略选择器 */}
        <div className="strategy-selector glass">
          <div className="strategy-header">
            <Layers size={18} />
            <span>切块策略</span>
          </div>
          <div className="strategy-options">
            {STRATEGY_OPTIONS.map((opt) => {
              const Icon = opt.icon
              return (
                <div
                  key={opt.value}
                  className={`strategy-card ${selectedStrategy === opt.value ? 'active' : ''}`}
                  onClick={() => setSelectedStrategy(opt.value)}
                >
                  <div className="strategy-card-header">
                    <Icon size={18} />
                    <span>{opt.label}</span>
                    {opt.tag && <span className="strategy-tag">{opt.tag}</span>}
                  </div>
                  <p className="strategy-desc">{opt.desc}</p>
                </div>
              )
            })}
          </div>
        </div>

        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-input"
            accept=".pdf,.docx,.doc,.txt,.md,.xlsx,.xls,.pptx,.ppt"
            onChange={handleFileSelect}
            multiple
            hidden
          />
          <label htmlFor="file-input" className="upload-label">
            <div className="upload-icon">
              <Upload size={32} />
            </div>
            <h3>拖拽文件到此处</h3>
            <p>或点击选择文件</p>
            <span className="upload-hint">
              支持 PDF、Word、Excel、PPT、TXT、Markdown 格式
              <br />
              当前策略: <strong>{getStrategyLabel(selectedStrategy)}</strong>
            </span>
          </label>
        </div>

        {/* 上传进度列表 */}
        {uploadTasks.length > 0 && (
          <div className="upload-tasks glass">
            <div className="upload-tasks-header">
              <span>上传任务 ({uploadTasks.filter(t => t.status === UPLOAD_STATUS.SUCCESS).length}/{uploadTasks.length})</span>
              {uploadTasks.filter(t => t.status !== UPLOAD_STATUS.UPLOADING && t.status !== UPLOAD_STATUS.PENDING).length > 0 && (
                <button className="clear-tasks-btn" onClick={clearCompletedTasks}>清除已完成</button>
              )}
            </div>
            <div className="upload-tasks-list">
              {uploadTasks.map(task => (
                <div key={task.id} className={`upload-task-item ${task.status}`}>
                  <div className="upload-task-icon">
                    {task.status === UPLOAD_STATUS.PENDING && <Clock size={16} />}
                    {task.status === UPLOAD_STATUS.UPLOADING && <Loader2 size={16} className="spin" />}
                    {task.status === UPLOAD_STATUS.SUCCESS && <CheckCircle size={16} />}
                    {task.status === UPLOAD_STATUS.FAILED && <XCircle size={16} />}
                  </div>
                  <div className="upload-task-info">
                    <span className="upload-task-name">{task.name}</span>
                    <span className="upload-task-status">
                      {task.status === UPLOAD_STATUS.PENDING && '等待中'}
                      {task.status === UPLOAD_STATUS.UPLOADING && '上传中...'}
                      {task.status === UPLOAD_STATUS.SUCCESS && '上传成功'}
                      {task.status === UPLOAD_STATUS.FAILED && (task.error || '上传失败')}
                    </span>
                  </div>
                  {(task.status === UPLOAD_STATUS.SUCCESS || task.status === UPLOAD_STATUS.FAILED) && (
                    <button className="dismiss-task-btn" onClick={() => dismissTask(task.id)}>
                      <XCircle size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="documents-section">
          <h2>已上传文档 ({documents.length})</h2>

          {loading ? (
            <div className="loading-state">
              <RefreshCw size={24} className="spin" />
              <span>加载中...</span>
            </div>
          ) : documents.length === 0 ? (
            <div className="empty-docs">
              <FileText size={48} />
              <p>暂无已上传的文档</p>
            </div>
          ) : (
            <div className="documents-grid">
              {documents.map((doc, index) => (
                <div key={doc.id} className="document-card glass" style={{ animationDelay: `${index * 0.1}s` }}>
                  <div className="doc-icon">
                    <FileText size={28} />
                  </div>
                  <div className="doc-info">
                    <h4 className="doc-name" title={doc.document_name}>{doc.document_name}</h4>
                    <div className="doc-meta-row">
                      <div className="doc-meta">
                        {getStatusIcon(doc.status)}
                        <span className="doc-status">{getStatusText(doc.status)}</span>
                      </div>
                      {doc.chunk_strategy && (
                        <span className="doc-strategy-badge" title={doc.chunk_strategy}>
                          {getStrategyLabel(doc.chunk_strategy)}
                        </span>
                      )}
                    </div>
                  </div>
                  <button className="delete-btn" onClick={() => handleDelete(doc.id)}>
                    <Trash2 size={18} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DocumentsPage