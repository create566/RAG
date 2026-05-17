import { createContext, useContext, useState, useEffect } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(localStorage.getItem('token'))
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const storedToken = localStorage.getItem('token')
    const storedUser = localStorage.getItem('user')
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
    }
    setLoading(false)
  }, [])

  const login = async (username, password) => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || '登录失败')
    }

    const data = await response.json()
    localStorage.setItem('token', data.access_token)
    localStorage.setItem('user_id', data.user_id)
    localStorage.setItem('username', data.username)
    setToken(data.access_token)
    setUser({ user_id: data.user_id, username: data.username })
    return data
  }

  const register = async (username, password, email = '', phone = '', department = '', position = '', fullName = '') => {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, email, phone, department, position, full_name: fullName })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || '注册失败')
    }

    const data = await response.json()
    localStorage.setItem('token', data.access_token)
    localStorage.setItem('user_id', data.user_id)
    localStorage.setItem('username', data.username)
    setToken(data.access_token)
    setUser({ user_id: data.user_id, username: data.username })
    return data
  }

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user_id')
    localStorage.removeItem('username')
    setToken(null)
    setUser(null)
  }

  const value = {
    user,
    token,
    loading,
    login,
    register,
    logout,
    isAuthenticated: !!token
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}