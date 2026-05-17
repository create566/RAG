import { useState, useEffect } from 'react'
import { Network, RefreshCw, Search, ChevronRight, FileText, Hash, Layers } from 'lucide-react'
import { graphApi, documentApi } from '../api/services'
import './GraphPage.css'

function GraphPage() {
  const [documents, setDocuments] = useState([])
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [graphData, setGraphData] = useState(null)
  const [selectedChapter, setSelectedChapter] = useState(null)
  const [chapterContent, setChapterContent] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingChapter, setLoadingChapter] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    try {
      const data = await documentApi.list()
      if (data.success) {
        setDocuments(data.documents || [])
      }
    } catch (error) {
      console.error('Failed to load documents:', error)
    }
  }

  const loadGraphData = async (docId) => {
    setLoading(true)
    setSelectedChapter(null)
    setChapterContent(null)
    try {
      const res = await graphApi.getDocumentStructure(docId)
      if (res.success) {
        setGraphData(res.structure)
        setSelectedDoc(docId)
      }
    } catch (error) {
      console.error('Failed to load graph:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadChapterContent = async (chapterTitle) => {
    if (!selectedDoc) return
    setLoadingChapter(true)
    try {
      const res = await graphApi.getChapterParagraphs(selectedDoc, chapterTitle)
      if (res.success && res.result) {
        setChapterContent(res.result)
        setSelectedChapter(chapterTitle)
      }
    } catch (error) {
      console.error('Failed to load chapter:', error)
    } finally {
      setLoadingChapter(false)
    }
  }

  const cypherQuery = async () => {
    if (!searchQuery.trim()) return
    setLoading(true)
    try {
      const res = await graphApi.cypher(searchQuery)
      if (res.success) {
        setGraphData({ customQuery: true, results: res.results })
      }
    } catch (error) {
      console.error('Cypher query failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="graph-page">
      <header className="page-header glass">
        <div className="header-left">
          <h1>知识图谱</h1>
          <span className="header-subtitle">Neo4j 图数据库可视化</span>
        </div>
      </header>

      <div className="graph-container">
        {/* 左侧：文档列表 */}
        <div className="doc-panel glass">
          <h3>
            <FileText size={16} />
            文档列表
          </h3>
          <div className="doc-list">
            {documents.length === 0 ? (
              <div className="empty-list">暂无文档</div>
            ) : (
              documents.map((doc) => (
                <div
                  key={doc.id}
                  className={`doc-item ${selectedDoc === doc.id ? 'active' : ''}`}
                  onClick={() => loadGraphData(doc.id)}
                >
                  <FileText size={14} />
                  <span className="doc-name">{doc.document_name}</span>
                  <ChevronRight size={14} />
                </div>
              ))
            )}
          </div>
        </div>

        {/* 中间：图谱可视化 */}
        <div className="graph-panel">
          {!graphData ? (
            <div className="empty-graph">
              <Network size={64} />
              <h3>选择文档查看图谱</h3>
              <p>点击左侧文档，查看其章节结构和关系</p>
            </div>
          ) : loading ? (
            <div className="loading-graph">
              <RefreshCw size={32} className="spin" />
              <span>加载中...</span>
            </div>
          ) : graphData.customQuery ? (
            <div className="query-results">
              <h3>查询结果 ({graphData.results?.length || 0})</h3>
              <pre>{JSON.stringify(graphData.results, null, 2)}</pre>
            </div>
          ) : (
            <div className="graph-content">
              <div className="graph-header">
                <h2>{graphData.document_title || '文档图谱'}</h2>
                <span className="graph-stat">
                  <Hash size={14} />
                  文档ID: {selectedDoc}
                </span>
              </div>

              {/* 统计信息 */}
              <div className="graph-stats">
                <div className="stat-card">
                  <Layers size={20} />
                  <div className="stat-info">
                    <span className="stat-value">{graphData.chapters?.length || 0}</span>
                    <span className="stat-label">章节</span>
                  </div>
                </div>
                <div className="stat-card">
                  <FileText size={20} />
                  <div className="stat-info">
                    <span className="stat-value">{graphData.paragraph_count || 0}</span>
                    <span className="stat-label">段落</span>
                  </div>
                </div>
              </div>

              {/* 章节列表 - 可点击 */}
              <div className="chapters-section">
                <h3>章节结构（点击查看内容）</h3>
                <div className="chapters-list">
                  {graphData.chapters?.map((chapter, index) => (
                    <div
                      key={index}
                      className={`chapter-item ${selectedChapter === chapter ? 'active' : ''}`}
                      onClick={() => loadChapterContent(chapter)}
                    >
                      <div className="chapter-badge">{index + 1}</div>
                      <span className="chapter-title">{chapter}</span>
                      {selectedChapter === chapter && <ChevronRight size={14} className="chapter-arrow" />}
                    </div>
                  ))}
                </div>
              </div>

              {/* 段落内容 */}
              {chapterContent && (
                <div className="chapter-content-section">
                  <h3>
                    <FileText size={16} />
                    {chapterContent.chapter_title || selectedChapter} - 段落内容
                  </h3>
                  <div className="chapter-content">
                    {chapterContent.paragraphs && chapterContent.paragraphs.length > 0 ? (
                      chapterContent.paragraphs.map((para, idx) => (
                        <p key={idx} className="paragraph-item">{para}</p>
                      ))
                    ) : chapterContent.chapter_content ? (
                      <div className="full-content">{chapterContent.chapter_content}</div>
                    ) : (
                      <p className="no-content">暂无段落内容</p>
                    )}
                  </div>
                </div>
              )}

              {/* 图谱可视化 */}
              <div className="visual-graph">
                <h3>图谱视图</h3>
                <div className="graph-canvas">
                  <div className="graph-node root">
                    <FileText size={16} />
                    <span>{graphData.document_title || 'Document'}</span>
                  </div>
                  {graphData.chapters?.map((chapter, index) => (
                    <div key={index} className="graph-node child">
                      <div className="node-line" />
                      <div className="node-badge">{index + 1}</div>
                      <span>{chapter}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 右侧：Cypher查询 */}
        <div className="query-panel glass">
          <h3>
            <Search size={16} />
            Cypher 查询
          </h3>
          <textarea
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="MATCH (d:Document)-[:CONTAINS]->(c:Chapter) RETURN d.title, c.title"
            rows={4}
          />
          <button className="query-btn" onClick={cypherQuery} disabled={loading}>
            执行查询
          </button>

          <div className="query-examples">
            <h4>示例查询</h4>
            <button onClick={() => setSearchQuery('MATCH (d:Document) RETURN d')}>
              查看所有文档
            </button>
            <button onClick={() => setSearchQuery('MATCH (d:Document)-[:CONTAINS]->(c:Chapter) RETURN d,c')}>
              文档-章节关系
            </button>
            <button onClick={() => setSearchQuery('MATCH (c:Chapter)-[:CONTAINS]->(p:Paragraph) RETURN c.title, p.content LIMIT 20')}>
              章节-段落内容
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default GraphPage