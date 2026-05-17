const API_BASE = '/api'

// 聊天相关API
export const chatApi = {
  chat: async (question, conversationId = null, chatMode = 'auto', userId = null) => {
    const response = await fetch(`${API_BASE}/chat/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        chat_mode: chatMode,
        user_id: userId || parseInt(localStorage.getItem('user_id'))
      })
    })
    return response.json()
  },

  streamChat: (question, conversationId = null, chatMode = 'auto', userId = null) => {
    return fetch(`${API_BASE}/chat/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        chat_mode: chatMode,
        user_id: userId || parseInt(localStorage.getItem('user_id'))
      })
    })
  },

  getHistory: async (conversationId) => {
    const response = await fetch(`${API_BASE}/chat/conversation/${conversationId}/history`)
    return response.json()
  }
}

// 文档相关API
export const documentApi = {
  upload: async (file, userId = null, documentName = null, chunkStrategy = null) => {
    const formData = new FormData()
    formData.append('file', file)
    if (documentName) formData.append('document_name', documentName)
    if (userId != null && userId !== 0) formData.append('user_id', userId)
    if (chunkStrategy) formData.append('chunk_strategy', chunkStrategy)

    const response = await fetch(`${API_BASE}/document/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: formData
    })
    return response.json()
  },

  list: async (userId = null) => {
    const url = userId ? `${API_BASE}/document/list?user_id=${userId}` : `${API_BASE}/document/list`
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    return response.json()
  },

  delete: async (documentId) => {
    const response = await fetch(`${API_BASE}/document/${documentId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    return response.json()
  }
}

// 图谱相关API
export const graphApi = {
  getDocumentStructure: async (documentId) => {
    const response = await fetch(`${API_BASE}/chat/graph/document/${documentId}`)
    return response.json()
  },

  getChapter: async (documentId, sectionHint) => {
    const response = await fetch(
      `${API_BASE}/chat/graph/document/${documentId}/chapter?section_hint=${encodeURIComponent(sectionHint)}`
    )
    return response.json()
  },

  getChapterParagraphs: async (documentId, chapterTitle) => {
    const response = await fetch(
      `${API_BASE}/chat/graph/document/${documentId}/chapter?section_hint=${encodeURIComponent(chapterTitle)}`
    )
    return response.json()
  },

  cypher: async (query) => {
    const response = await fetch(`${API_BASE}/chat/graph/cypher?query=${encodeURIComponent(query)}`)
    return response.json()
  }
}