import { apiClient, API_BASE } from './client'

const userId = () => parseInt(localStorage.getItem('user_id'))

// 聊天
export const chatApi = {
  chat: async (question, conversationId = null, chatMode = 'auto') => {
    const res = await apiClient('/chat/chat', {
      method: 'POST',
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        chat_mode: chatMode,
        user_id: userId(),
      }),
    })
    return res.json()
  },

  streamChat: (question, conversationId = null, chatMode = 'auto') => {
    return apiClient('/chat/chat/stream', {
      method: 'POST',
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        chat_mode: chatMode,
        user_id: userId(),
      }),
    })
  },

  getHistory: async (conversationId) => {
    const res = await apiClient(`/chat/conversation/${conversationId}/history`)
    return res.json()
  },
}

// 文档
export const documentApi = {
  upload: async (file, uid, documentName = null, chunkStrategy = null) => {
    const formData = new FormData()
    formData.append('file', file)
    if (documentName) formData.append('document_name', documentName)
    if (uid != null && uid !== 0) formData.append('user_id', uid)
    if (chunkStrategy) formData.append('chunk_strategy', chunkStrategy)

    const res = await apiClient('/document/upload', {
      method: 'POST',
      body: formData,
    })
    return res.json()
  },

  list: async (uid = null) => {
    const url = uid ? `/document/list?user_id=${uid}` : '/document/list'
    const res = await apiClient(url)
    return res.json()
  },

  delete: async (documentId) => {
    const res = await apiClient(`/document/${documentId}`, { method: 'DELETE' })
    return res.json()
  },
}

// 图谱
export const graphApi = {
  getDocumentStructure: async (documentId) => {
    const res = await apiClient(`/chat/graph/document/${documentId}`)
    return res.json()
  },

  getChapter: async (documentId, sectionHint) => {
    const res = await apiClient(
      `/chat/graph/document/${documentId}/chapter?section_hint=${encodeURIComponent(sectionHint)}`
    )
    return res.json()
  },

  getChapterParagraphs: async (documentId, chapterTitle) => {
    const res = await apiClient(
      `/chat/graph/document/${documentId}/chapter?section_hint=${encodeURIComponent(chapterTitle)}`
    )
    return res.json()
  },

  cypher: async (query) => {
    const res = await apiClient(`/chat/graph/cypher?query=${encodeURIComponent(query)}`)
    return res.json()
  },
}
