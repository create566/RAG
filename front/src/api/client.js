/**
 * 统一 API 客户端 — 自动注入 Authorization header + 401 处理
 */
const API_BASE = '/api'

async function apiClient(endpoint, options = {}) {
  const token = localStorage.getItem('token')

  const headers = {
    ...options.headers,
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  // FormData 不设置 Content-Type（浏览器会自动设 multipart/form-data）
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  })

  if (response.status === 401) {
    localStorage.clear()
    window.location.href = '/login'
    throw new Error('Authentication required')
  }

  return response
}

export { apiClient, API_BASE }
