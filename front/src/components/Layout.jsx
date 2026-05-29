import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom'
import { MessageSquare, FileText, Network, Menu, Plus, Trash2, ChevronLeft, ChevronRight } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { apiClient } from '../api/client'
import './Layout.css'

const CHAT_API = '/chat'

const navItems = [
  { path: '/chat', label: '智能对话', icon: MessageSquare },
  { path: '/documents', label: '文档管理', icon: FileText },
  { path: '/graph', label: '知识图谱', icon: Network },
]

function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [conversations, setConversations] = useState([])
  const [loadingConv, setLoadingConv] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const { user } = useAuth()

  const userId = user?.user_id || parseInt(localStorage.getItem('user_id'))
  const activeConvId = new URLSearchParams(location.search).get('conv_id')

  const loadConversations = async () => {
    if (!userId) return
    setLoadingConv(true)
    try {
      const res = await apiClient(`${CHAT_API}/conversations?user_id=${userId}`)
      const data = await res.json()
      if (data.success) setConversations(data.date_groups || [])
    } catch (e) { console.error(e) }
    finally { setLoadingConv(false) }
  }

  useEffect(() => { loadConversations() }, [userId])

  // 每 30 秒刷新
  useEffect(() => {
    const interval = setInterval(loadConversations, 30000)
    return () => clearInterval(interval)
  }, [userId])

  const handleNewChat = () => {
    navigate('/chat?new=1')
  }

  const handleSelectConv = (convId) => {
    navigate(`/chat?conv_id=${convId}`)
  }

  const handleDeleteConv = async (convId, e) => {
    e.stopPropagation()
    if (!confirm('删除该会话？')) return
    try {
      await apiClient(`${CHAT_API}/conversations/${convId}`, { method: 'DELETE' })
      if (convId === activeConvId) navigate('/chat')
      loadConversations()
    } catch (e) { console.error(e) }
  }

  const isChatPage = location.pathname === '/chat'

  return (
    <div className="layout">
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">
              <svg viewBox="0 0 40 40" fill="none">
                <defs>
                  <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#FF6B6B"/>
                    <stop offset="33%" stopColor="#4ECDC4"/>
                    <stop offset="66%" stopColor="#A855F7"/>
                    <stop offset="100%" stopColor="#3B82F6"/>
                  </linearGradient>
                </defs>
                <circle cx="20" cy="20" r="18" fill="url(#logoGradient)" fillOpacity="0.15" stroke="url(#logoGradient)" strokeWidth="1.5"/>
                <path d="M20 8 L20 32 M12 14 L28 14 M12 26 L28 26 M14 20 L26 20" stroke="url(#logoGradient)" strokeWidth="1.5" strokeLinecap="round"/>
                <circle cx="20" cy="20" r="4" fill="url(#logoGradient)"/>
              </svg>
            </div>
            {sidebarOpen && <span className="logo-text">RAG智能检索</span>}
          </div>
          <button className="toggle-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <ChevronLeft size={20} /> : <Menu size={20} />}
          </button>
        </div>

        <nav className="nav-menu">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <item.icon size={22} />
              {sidebarOpen && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* 历史会话列表 — 仅在聊天页显示 */}
        {sidebarOpen && isChatPage && (
          <div className="conv-section">
            {/* 新建对话按钮 — 豆包风格 */}
            <div className="conv-new-chat-wrap">
              <button className="conv-new-chat-btn" onClick={handleNewChat}>
                <Plus size={18} />
                <span>新建对话</span>
              </button>
            </div>

            <div className="conv-section-header">
              <span className="conv-section-title">历史会话</span>
            </div>

            <div className="conv-list">
              {loadingConv && conversations.length === 0 ? (
                <div className="conv-list-hint">加载中...</div>
              ) : conversations.length === 0 ? (
                <div className="conv-list-hint">暂无对话</div>
              ) : (
                conversations.map((group, gi) => (
                  <div key={gi} className="conv-date-group">
                    <div className="conv-date-label">{group.date}</div>
                    {group.conversations.map(conv => (
                      <div
                        key={conv.id}
                        className={`conv-item ${conv.id === activeConvId ? 'active' : ''}`}
                        onClick={() => handleSelectConv(conv.id)}
                      >
                        <MessageSquare size={15} className="conv-item-icon" />
                        <span className="conv-item-title">{conv.title || '新对话'}</span>
                        <button
                          className="conv-item-del"
                          onClick={(e) => handleDeleteConv(conv.id, e)}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    ))}
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {!sidebarOpen && isChatPage && (
          <div className="conv-collapsed-hint" onClick={() => setSidebarOpen(true)}>
            <ChevronRight size={18} />
          </div>
        )}

        <div className="sidebar-footer">
          {sidebarOpen && (
            <div className="status-indicator">
              <span className="status-dot"></span>
              <span>系统正常</span>
            </div>
          )}
        </div>
      </aside>

      <main className="main-content gradient-mesh">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout