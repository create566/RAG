import { useState, useRef, useEffect } from 'react'
import { Send, Square, Trash2, Copy, User, Bot, LogOut, Plus } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useSearchParams } from 'react-router-dom'
import { apiClient } from '../api/client'
import './ChatPage.css'

const CHAT_API = '/chat'

function getConvStorageKey(userId) {
  return `super_chat_conv_id_${userId || 0}`
}

function ChatPage() {
  const { user, logout } = useAuth()
  const userId = user?.user_id || parseInt(localStorage.getItem('user_id'))
  const CONV_ID_KEY = getConvStorageKey(userId)
  const [searchParams] = useSearchParams()
  const convIdFromUrl = searchParams.get('conv_id')
  const isNewChat = searchParams.get('new') === '1'

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [currentAnswer, setCurrentAnswer] = useState('')
  const [conversationId, setConversationId] = useState(null)
  const [loadedConvId, setLoadedConvId] = useState(null)

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)

  // URL 参数变化时切换会话
  useEffect(() => {
    // 新建对话标记 → 清空一切
    if (isNewChat) {
      handleNewChat()
      return
    }

    if (convIdFromUrl && convIdFromUrl !== loadedConvId) {
      setConversationId(convIdFromUrl)
      setLoadedConvId(convIdFromUrl)
      loadConversation(convIdFromUrl)
    } else if (!convIdFromUrl && !loadedConvId) {
      // 首次加载无 URL 参数：恢复上次会话
      const savedConvId = localStorage.getItem(CONV_ID_KEY)
      if (savedConvId) {
        setConversationId(savedConvId)
        setLoadedConvId(savedConvId)
        loadConversation(savedConvId)
      }
    }
  }, [convIdFromUrl, isNewChat])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentAnswer])

  const loadConversation = async (convId) => {
    setCurrentAnswer('')
    try {
      const res = await apiClient(`${CHAT_API}/conversations/${convId}/messages`)
      const data = await res.json()
      if (data.success && data.messages) {
        setMessages(data.messages.map(m => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: new Date(m.created_at || Date.now())
        })))
      }
    } catch (e) {
      console.error('loadConversation error:', e)
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setConversationId(null)
    setLoadedConvId(null)
    setCurrentAnswer('')
    localStorage.removeItem(CONV_ID_KEY)
    window.history.replaceState(null, '', '/chat')
  }

  const handleStreamResponse = async () => {
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')
    setLoading(true)
    setStreaming(true)
    setCurrentAnswer('')

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])

    const isNewConv = !conversationId

    try {
      const controller = new AbortController()
      abortRef.current = controller

      const response = await apiClient(`${CHAT_API}/chat/stream`, {
        method: 'POST',
        body: JSON.stringify({
          question,
          conversation_id: conversationId || undefined,
          chat_mode: 'AUTO_DOCUMENT',
          user_id: userId
        }),
        signal: controller.signal
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullAnswer = ''
      let newConvId = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.answer) {
                fullAnswer += data.answer
                setCurrentAnswer(fullAnswer)
              }
              if (data.conversation_id) {
                newConvId = data.conversation_id
              }
            } catch (e) {}
          }
        }
      }

      if (fullAnswer) {
        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: fullAnswer,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, assistantMessage])
      }

      if (newConvId) {
        setConversationId(newConvId)
        setLoadedConvId(newConvId)
        localStorage.setItem(CONV_ID_KEY, newConvId)
        window.history.replaceState(null, '', `/chat?conv_id=${newConvId}`)
      }

      setCurrentAnswer('')

    } catch (error) {
      if (error.name !== 'AbortError') {
        setCurrentAnswer('请求失败: ' + error.message)
      }
    } finally {
      setLoading(false)
      setStreaming(false)
      abortRef.current = null
      inputRef.current?.focus()
    }
  }

  const handleStop = () => {
    abortRef.current?.abort()
    setStreaming(false)
    setLoading(false)
  }

  const handleClear = () => {
    // 清空消息 → 导航到新建对话
    localStorage.removeItem(CONV_ID_KEY)
    window.location.href = '/chat?new=1'
  }

  const copyMessage = (content) => {
    navigator.clipboard.writeText(content)
  }

  return (
    <div className="chat-page">
      <header className="chat-header glass">
        <div className="header-left">
          <h1>智能对话</h1>
          <span className="header-subtitle">AI自动判断意图 · RAG/Agent/工具调用</span>
        </div>
        <div className="header-actions">
          <span className="username-display">{user?.username || localStorage.getItem('username')}</span>
          <button className="icon-btn" onClick={handleClear} title="新对话">
            <Plus size={20} />
          </button>
          <button className="icon-btn" onClick={logout} title="退出登录">
            <LogOut size={20} />
          </button>
        </div>
      </header>

      <div className="messages-container">
        {messages.length === 0 && !streaming && (
          <div className="empty-state">
            <div className="empty-icon">
              <svg viewBox="0 0 80 80" fill="none">
                <circle cx="40" cy="40" r="36" stroke="url(#emptyGradient)" strokeWidth="2" strokeDasharray="4 4"/>
                <path d="M25 35h30M25 45h20" stroke="#0ea5e9" strokeWidth="2" strokeLinecap="round"/>
                <defs>
                  <linearGradient id="emptyGradient" x1="0" y1="0" x2="80" y2="80">
                    <stop offset="0%" stopColor="#0ea5e9"/>
                    <stop offset="100%" stopColor="#10b981"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <h2>开始新对话</h2>
            <p>输入您的问题，AI助手将为您解答</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            <div className="message-content">
              <div className="message-bubble">
                <p>{msg.content}</p>
                <div className="message-actions">
                  <button onClick={() => copyMessage(msg.content)} className="action-btn">
                    <Copy size={14} />
                  </button>
                </div>
              </div>
              <span className="message-time">
                {msg.timestamp instanceof Date && !isNaN(msg.timestamp)
                  ? msg.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
                  : ''}
              </span>
            </div>
          </div>
        ))}

        {streaming && currentAnswer && (
          <div className="message assistant streaming">
            <div className="message-avatar"><Bot size={20} /></div>
            <div className="message-content">
              <div className="message-bubble">
                <p>{currentAnswer}<span className="cursor">|</span></p>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-container glass">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleStreamResponse()
              }
            }}
            placeholder="输入您的问题... (Enter发送, Shift+Enter换行)"
            rows={1}
            disabled={loading}
          />
          <div className="input-actions">
            {loading ? (
              <button className="stop-btn" onClick={handleStop}>
                <Square size={20} />
                <span>停止</span>
              </button>
            ) : (
              <button className="send-btn" onClick={handleStreamResponse} disabled={!input.trim()}>
                <Send size={20} />
                <span>发送</span>
              </button>
            )}
          </div>
        </div>
        <div className="input-hint">按 Enter 发送 · Shift + Enter 换行</div>
      </div>
    </div>
  )
}

export default ChatPage