import { useState, useEffect, useRef, ChangeEvent, DragEvent } from 'react'

interface NLPResult {
  texto_original: string
  intencion: string
  intencion_base: string
  confianza_intencion: number
  es_ambigua: boolean
  sentimiento: string
  confianza_sentimiento: number
  entidades: {
    sede: string
    equipo_insumo: string
    tipo_documento: string
    urgencia: string
  }
  distribucion_probabilidades?: Record<string, number>
}

interface TopClassItem {
  clase: string
  confianza: number
}

interface VisionResult {
  clase_imagen: string
  confianza_imagen: number
  es_confiable: boolean
  top3: TopClassItem[]
}

interface TriageDecision {
  prioridad: 'CRITICA' | 'MEDIA' | 'BAJA' | string
  cola_asignada: string
  accion_sugerida: string
  alerta_contradiccion: boolean
  motivo_decision: string
}

interface TicketTriageResponse {
  ticket_id: string
  fecha_procesamiento: string
  tiempo_procesamiento_ms: number
  nlp: NLPResult
  vision: VisionResult
  triaje: TriageDecision
}

interface BackendHealth {
  status: string
  app_env: string
  nlp_loaded: boolean
  vision_loaded: boolean
  nlp_threshold: number
  vision_threshold: number
}

interface Escenario {
  id: string
  nombre: string
  subtitulo: string
  categoria: string
  texto: string
  imagenUrl: string
  imagenNombre: string
  color: string
}

const ESCENARIOS: Escenario[] = [
  {
    id: 'esc_1',
    nombre: '1. Crítico Coherente',
    subtitulo: 'Falla térmica + Pantalla/Equipo',
    categoria: 'CRITICA',
    texto:
      'Urgente: La impresora térmica de Sede Sur tiene atasco de papel severo y no imprime los tickets, cabezal quemado',
    imagenUrl: '/ejemplos/falla_equipo.jpg',
    imagenNombre: 'falla_equipo.jpg',
    color: 'border-rose-500/40 text-rose-300 hover:bg-rose-950/40'
  },
  {
    id: 'esc_2',
    nombre: '2. Ambiguo + Doc. Borroso',
    subtitulo: 'Texto vago + Documento ilegible',
    categoria: 'MEDIA',
    texto: 'hola ayuda con esto porfa no pasa',
    imagenUrl: '/ejemplos/doc_borroso.jpg',
    imagenNombre: 'doc_borroso.jpg',
    color: 'border-amber-500/40 text-amber-300 hover:bg-amber-950/40'
  },
  {
    id: 'esc_3',
    nombre: '3. Consulta General',
    subtitulo: 'Requisitos de ayuno + Orden nítida',
    categoria: 'BAJA',
    texto:
      'Buenos días, quisiera consultar cuáles son los requisitos de ayuno y horarios para la toma de muestras',
    imagenUrl: '/ejemplos/doc_legible.jpg',
    imagenNombre: 'doc_legible.jpg',
    color: 'border-emerald-500/40 text-emerald-300 hover:bg-emerald-950/40'
  },
  {
    id: 'esc_4',
    nombre: '4. Contradicción: Falla vs OK',
    subtitulo: 'Reporta escáner roto + Insumo OK',
    categoria: 'CONTRADICCION',
    texto:
      'El escáner de código de barras está totalmente roto y no lee ninguna muestra en Consultorio 1',
    imagenUrl: '/ejemplos/insumo_ok.jpg',
    imagenNombre: 'insumo_ok.jpg',
    color: 'border-purple-500/40 text-purple-300 hover:bg-purple-950/40'
  },
  {
    id: 'esc_5',
    nombre: '5. Contradicción: Cita vs Ilegible',
    subtitulo: 'Pide adelantar cita + Orden borrosa',
    categoria: 'CONTRADICCION',
    texto:
      'Adjunto mi orden médica clara para adelantar cita urgente en Sede Central',
    imagenUrl: '/ejemplos/doc_borroso.jpg',
    imagenNombre: 'doc_borroso.jpg',
    color: 'border-purple-500/40 text-purple-300 hover:bg-purple-950/40'
  },
  {
    id: 'esc_6',
    nombre: '6. Gestión de Cita Válida',
    subtitulo: 'Reprogramar cita + Documento OK',
    categoria: 'BAJA',
    texto:
      'Buenas tardes, solicito por favor reprogramar mi cita de control en Sede Sur para la próxima semana',
    imagenUrl: '/ejemplos/doc_legible.jpg',
    imagenNombre: 'doc_legible.jpg',
    color: 'border-teal-500/40 text-teal-300 hover:bg-teal-950/40'
  }
]

export default function App() {
  // Configuración de API
  const defaultApiUrl =
    import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`
  const [apiUrl, setApiUrl] = useState<string>(defaultApiUrl)
  const [showConfigApi, setShowConfigApi] = useState<boolean>(false)
  const [tempApiUrl, setTempApiUrl] = useState<string>(apiUrl)

  // Estado del Backend
  const [health, setHealth] = useState<BackendHealth | null>(null)
  const [backendStatus, setBackendStatus] = useState<
    'checking' | 'online' | 'offline'
  >('checking')

  // Formulario de Ticket
  const [texto, setTexto] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState<boolean>(false)
  const [activeScenarioId, setActiveScenarioId] = useState<string | null>(null)

  // Inferencia y Resultados
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [evalResult, setEvalResult] = useState<TicketTriageResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [showRawJson, setShowRawJson] = useState<boolean>(false)
  const [copiedJson, setCopiedJson] = useState<boolean>(false)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // Comprobar estado de salud del backend
  const checkHealth = async (targetUrl = apiUrl) => {
    setBackendStatus('checking')
    try {
      const res = await fetch(`${targetUrl}/health`, { method: 'GET' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: BackendHealth = await res.json()
      setHealth(data)
      setBackendStatus('online')
    } catch {
      setHealth(null)
      setBackendStatus('offline')
    }
  }

  useEffect(() => {
    checkHealth(apiUrl)
  }, [apiUrl])

  // Manejo de archivos e imágenes
  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Por favor selecciona un archivo de imagen válido (JPG, PNG, WebP).')
      return
    }
    setSelectedFile(file)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setActiveScenarioId(null)
  }

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0])
    }
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  const handleClear = () => {
    setTexto('')
    setSelectedFile(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setActiveScenarioId(null)
    setEvalResult(null)
    setErrorMessage(null)
  }

  // Carga de escenarios precargados
  const loadScenario = async (esc: Escenario) => {
    setTexto(esc.texto)
    setActiveScenarioId(esc.id)
    setErrorMessage(null)

    try {
      const res = await fetch(esc.imagenUrl)
      if (!res.ok) throw new Error('No se pudo cargar la imagen de muestra')
      const blob = await res.blob()
      const file = new File([blob], esc.imagenNombre, { type: 'image/jpeg' })
      setSelectedFile(file)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(URL.createObjectURL(file))
    } catch (err: any) {
      console.error(err)
      setErrorMessage(`Error al cargar la imagen de muestra: ${err.message}`)
    }
  }

  // Enviar evaluación a la API
  const handleSubmit = async () => {
    if (!texto.trim()) {
      alert('Por favor ingresa el texto del ticket.')
      return
    }
    if (!selectedFile) {
      alert('Por favor adjunta una fotografía o captura de evidencia.')
      return
    }

    setIsLoading(true)
    setErrorMessage(null)

    try {
      const formData = new FormData()
      formData.append('texto', texto)
      formData.append('imagen', selectedFile)

      const res = await fetch(`${apiUrl}/triaje/evaluar-ticket`, {
        method: 'POST',
        body: formData
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(
          errorData.detail || `Error en la evaluación del ticket (HTTP ${res.status})`
        )
      }

      const data: TicketTriageResponse = await res.json()
      setEvalResult(data)
    } catch (err: any) {
      setErrorMessage(err.message || 'Error de conexión con el servicio de triaje.')
    } finally {
      setIsLoading(false)
    }
  }

  const copyToClipboard = () => {
    if (!evalResult) return
    navigator.clipboard.writeText(JSON.stringify(evalResult, null, 2))
    setCopiedJson(true)
    setTimeout(() => setCopiedJson(false), 2000)
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30 px-4 sm:px-6 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500/30 to-emerald-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400 shadow-lg shadow-teal-500/10">
              <i className="fa-solid fa-hospital-user text-xl"></i>
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-white flex items-center gap-2">
                Triaje Multimodal Clínico
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30">
                  IA v1.0
                </span>
              </h1>
              <p className="text-xs text-slate-400 hidden sm:block">
                Clasificación Multicanal con NLP (TF-IDF + Calibrated LinearSVC) y Visión (YOLOv8-cls)
              </p>
            </div>
          </div>

          {/* Connection Status Badge */}
          <div className="flex items-center space-x-2">
            <div
              className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium border ${
                backendStatus === 'online'
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : backendStatus === 'checking'
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
              }`}
            >
              <i
                className={`fa-solid fa-circle text-[8px] ${
                  backendStatus === 'online'
                    ? 'animate-pulse text-emerald-400'
                    : backendStatus === 'checking'
                    ? 'animate-spin fa-spinner'
                    : 'text-rose-400'
                }`}
              ></i>
              <span>
                {backendStatus === 'online'
                  ? 'Backend Conectado'
                  : backendStatus === 'checking'
                  ? 'Verificando...'
                  : 'Backend Desconectado'}
              </span>
            </div>

            <button
              onClick={() => {
                setTempApiUrl(apiUrl)
                setShowConfigApi(!showConfigApi)
              }}
              title="Configurar IP del Servidor"
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs transition"
            >
              <i className="fa-solid fa-sliders"></i>
            </button>
          </div>
        </div>

        {/* Server IP Override Bar */}
        {showConfigApi && (
          <div className="max-w-7xl mx-auto mt-3 p-3 bg-slate-800/80 border border-slate-700 rounded-lg flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 flex-1 min-w-[260px]">
              <i className="fa-solid fa-network-wired text-teal-400"></i>
              <span className="text-slate-300 font-medium">URL de la API:</span>
              <input
                type="text"
                value={tempApiUrl}
                onChange={(e) => setTempApiUrl(e.target.value)}
                placeholder="http://192.168.137.1:8000"
                className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 flex-1 font-mono text-xs focus:outline-none focus:border-teal-500"
              />
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setApiUrl(tempApiUrl)
                  setShowConfigApi(false)
                }}
                className="px-3 py-1 bg-teal-600 hover:bg-teal-500 text-white rounded font-medium transition"
              >
                Aplicar
              </button>
              <button
                onClick={() => {
                  setTempApiUrl(defaultApiUrl)
                  setApiUrl(defaultApiUrl)
                  setShowConfigApi(false)
                }}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition"
              >
                Auto-detectar
              </button>
            </div>
          </div>
        )}
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* ========================================================= */}
        {/* PANEL IZQUIERDO: Entrada de Ticket y Escenarios (5 cols) */}
        {/* ========================================================= */}
        <section className="lg:col-span-5 space-y-5">
          {/* 6 Escenarios Rápidos */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <i className="fa-solid fa-bolt text-amber-400"></i>
                Escenarios de Prueba (1 Clic)
              </span>
              <span className="text-[10px] text-slate-500 font-mono">6 Casos</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {ESCENARIOS.map((esc) => {
                const isActive = activeScenarioId === esc.id
                return (
                  <button
                    key={esc.id}
                    onClick={() => loadScenario(esc)}
                    className={`text-left p-2.5 rounded-xl border transition-all text-xs flex flex-col justify-between ${
                      isActive
                        ? 'bg-teal-950/60 border-teal-500 text-white ring-1 ring-teal-500/40 shadow-md'
                        : `${esc.color} bg-slate-950/40 hover:border-slate-600`
                    }`}
                  >
                    <div className="font-semibold truncate">{esc.nombre}</div>
                    <div className="text-[10px] text-slate-400 truncate mt-0.5">
                      {esc.subtitulo}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Formulario de Entrada */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <i className="fa-solid fa-message text-teal-400"></i>
                Mensaje del Ticket
              </label>
              <span className="text-[11px] text-slate-500 font-mono">
                {texto.length} caracteres
              </span>
            </div>

            <textarea
              value={texto}
              onChange={(e) => setTexto(e.target.value)}
              placeholder="Escribe el reporte clínico o motivo del ticket aquí... (ej: 'La impresora térmica bota humo y está trabada en Sede Central')"
              rows={4}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition resize-none font-sans"
            />

            {/* Subida de Imagen con Soporte de Cámara Móvil */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <i className="fa-solid fa-camera text-teal-400"></i>
                  Evidencia Fotográfica
                </label>
                {previewUrl && (
                  <button
                    onClick={() => {
                      setSelectedFile(null)
                      if (previewUrl) URL.revokeObjectURL(previewUrl)
                      setPreviewUrl(null)
                    }}
                    className="text-[11px] text-rose-400 hover:text-rose-300 transition flex items-center gap-1"
                  >
                    <i className="fa-solid fa-trash-can"></i> Quitar
                  </button>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleInputChange}
                className="hidden"
              />

              {!previewUrl ? (
                <div
                  onDragOver={(e) => {
                    e.preventDefault()
                    setIsDragging(true)
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition flex flex-col items-center justify-center space-y-2 ${
                    isDragging
                      ? 'border-teal-400 bg-teal-500/10'
                      : 'border-slate-800 hover:border-slate-700 bg-slate-950/40 hover:bg-slate-950/70'
                  }`}
                >
                  <div className="w-12 h-12 rounded-full bg-slate-800/80 flex items-center justify-center text-slate-400">
                    <i className="fa-solid fa-cloud-arrow-up text-xl"></i>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-slate-200">
                      Toca para tomar foto o elegir archivo
                    </p>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Soporta cámara móvil directa o archivos JPG, PNG, WebP
                    </p>
                  </div>
                </div>
              ) : (
                <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex flex-col items-center p-2">
                  <img
                    src={previewUrl}
                    alt="Previsualización"
                    className="max-h-48 w-auto rounded-lg object-contain shadow-md"
                  />
                  <div className="mt-2 text-center">
                    <span className="text-[11px] text-slate-400 font-mono">
                      {selectedFile?.name} ({(selectedFile?.size || 0) / 1024 > 1024
                        ? `${((selectedFile?.size || 0) / (1024 * 1024)).toFixed(1)} MB`
                        : `${((selectedFile?.size || 0) / 1024).toFixed(0)} KB`})
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Error Banner */}
            {errorMessage && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
                <i className="fa-solid fa-circle-exclamation text-rose-400 text-sm"></i>
                <div className="flex-1">{errorMessage}</div>
              </div>
            )}

            {/* Botones de Acción */}
            <div className="flex gap-2 pt-2">
              <button
                onClick={handleSubmit}
                disabled={isLoading || !texto.trim() || !selectedFile}
                className={`flex-1 py-3 px-4 rounded-xl font-bold text-sm shadow-lg flex items-center justify-center gap-2 transition ${
                  isLoading || !texto.trim() || !selectedFile
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                    : 'bg-gradient-to-r from-teal-500 to-emerald-600 hover:from-teal-400 hover:to-emerald-500 text-white shadow-teal-500/20'
                }`}
              >
                {isLoading ? (
                  <>
                    <i className="fa-solid fa-spinner fa-spin"></i>
                    Evaluando Triaje...
                  </>
                ) : (
                  <>
                    <i className="fa-solid fa-wand-magic-sparkles"></i>
                    Evaluar Ticket Multimodal
                  </>
                )}
              </button>

              <button
                onClick={handleClear}
                title="Limpiar campos"
                className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl border border-slate-700 transition"
              >
                <i className="fa-solid fa-rotate-left"></i>
              </button>
            </div>
          </div>
        </section>

        {/* ========================================================= */}
        {/* PANEL DERECHO: Diagnóstico y Decisión de Triaje (7 cols) */}
        {/* ========================================================= */}
        <section className="lg:col-span-7 space-y-5">
          {!evalResult ? (
            <div className="h-full min-h-[400px] bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl flex flex-col items-center justify-center p-8 text-center">
              <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-500 text-2xl mb-4">
                <i className="fa-solid fa-clipboard-check"></i>
              </div>
              <h3 className="text-base font-semibold text-slate-300">
                Esperando Solicitud de Triaje
              </h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1">
                Selecciona uno de los 6 escenarios de prueba o escribe un mensaje con fotografía para obtener la evaluación multimodal en tiempo real.
              </p>
            </div>
          ) : (
            <div className="space-y-4 animate-fadeIn">
              {/* TARJETA 3: Decisión Final del Triaje (Destacada) */}
              <div
                className={`border rounded-2xl p-5 shadow-xl transition-all ${
                  evalResult.triaje.prioridad === 'CRITICA'
                    ? 'bg-rose-950/30 border-rose-500/50 shadow-rose-500/5'
                    : evalResult.triaje.prioridad === 'MEDIA'
                    ? 'bg-amber-950/30 border-amber-500/50 shadow-amber-500/5'
                    : 'bg-emerald-950/30 border-emerald-500/50 shadow-emerald-500/5'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase font-bold tracking-wider text-slate-400">
                      Resolución del Triaje
                    </span>
                    <span className="text-[11px] font-mono text-slate-500">
                      #{evalResult.ticket_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                    <i className="fa-solid fa-stopwatch text-teal-400"></i>
                    <span>{evalResult.tiempo_procesamiento_ms} ms</span>
                  </div>
                </div>

                {/* Banner de Contradicción si Aplica */}
                {evalResult.triaje.alerta_contradiccion && (
                  <div className="mt-3 p-3 rounded-xl bg-amber-500/20 border border-amber-500/40 text-amber-200 text-xs flex items-center gap-2.5 font-medium animate-pulse">
                    <i className="fa-solid fa-triangle-exclamation text-base text-amber-400"></i>
                    <span>
                      <strong>Alerta de Incoherencia Detectada:</strong> Existe una discrepancia entre lo descrito en el texto y la evidencia visual adjunta.
                    </span>
                  </div>
                )}

                {/* Prioridad y Cola Asignada */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                  <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80 flex items-center gap-3">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold shadow-md ${
                        evalResult.triaje.prioridad === 'CRITICA'
                          ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                          : evalResult.triaje.prioridad === 'MEDIA'
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      }`}
                    >
                      <i
                        className={`fa-solid ${
                          evalResult.triaje.prioridad === 'CRITICA'
                            ? 'fa-radiation animate-bounce'
                            : evalResult.triaje.prioridad === 'MEDIA'
                            ? 'fa-clock'
                            : 'fa-check'
                        }`}
                      ></i>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                        Prioridad Asignada
                      </div>
                      <div
                        className={`text-lg font-black tracking-tight ${
                          evalResult.triaje.prioridad === 'CRITICA'
                            ? 'text-rose-400'
                            : evalResult.triaje.prioridad === 'MEDIA'
                            ? 'text-amber-400'
                            : 'text-emerald-400'
                        }`}
                      >
                        {evalResult.triaje.prioridad}
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80 flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-teal-500/20 text-teal-400 border border-teal-500/30 flex items-center justify-center text-xl">
                      <i className="fa-solid fa-sitemap"></i>
                    </div>
                    <div className="truncate">
                      <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                        Cola Destino
                      </div>
                      <div className="text-sm font-bold text-white truncate">
                        {evalResult.triaje.cola_asignada}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Acción y Motivo */}
                <div className="mt-4 p-3 bg-slate-950/40 rounded-xl border border-slate-800/60 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] uppercase font-semibold text-slate-400">
                      Acción Sugerida:
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded font-mono font-semibold bg-slate-800 text-teal-300 border border-slate-700">
                      {evalResult.triaje.accion_sugerida}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    <strong>Motivo de decisión:</strong> {evalResult.triaje.motivo_decision}
                  </p>
                </div>
              </div>

              {/* TARJETAS DE CANALES MULTIMODALES (NLP + VISIÓN) */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* TARJETA 1: Módulo NLP */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                      <i className="fa-solid fa-brain text-teal-400"></i>
                      Diagnóstico NLP
                    </span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border ${
                        evalResult.nlp.sentimiento === 'NEGATIVO'
                          ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                          : evalResult.nlp.sentimiento === 'POSITIVO'
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : 'bg-slate-800 text-slate-300 border-slate-700'
                      }`}
                    >
                      {evalResult.nlp.sentimiento}
                    </span>
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-slate-400">Intención:</span>
                      <span className="font-bold text-white flex items-center gap-1">
                        {evalResult.nlp.intencion}
                        {evalResult.nlp.es_ambigua && (
                          <span className="text-[10px] text-amber-400 font-normal">
                            (&lt;65%)
                          </span>
                        )}
                      </span>
                    </div>

                    <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          evalResult.nlp.es_ambigua ? 'bg-amber-500' : 'bg-teal-500'
                        }`}
                        style={{
                          width: `${Math.min(
                            100,
                            evalResult.nlp.confianza_intencion * 100
                          )}%`
                        }}
                      ></div>
                    </div>
                    <div className="text-right text-[10px] text-slate-500 mt-1 font-mono">
                      Confianza: {(evalResult.nlp.confianza_intencion * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* Entidades Extraídas */}
                  <div className="pt-2 border-t border-slate-800/80">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                      Entidades NER Detectadas:
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        <strong className="text-slate-400">Sede:</strong>{' '}
                        {evalResult.nlp.entidades.sede}
                      </span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        <strong className="text-slate-400">Equipo:</strong>{' '}
                        {evalResult.nlp.entidades.equipo_insumo}
                      </span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        <strong className="text-slate-400">Doc:</strong>{' '}
                        {evalResult.nlp.entidades.tipo_documento}
                      </span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        <strong className="text-slate-400">Urg:</strong>{' '}
                        {evalResult.nlp.entidades.urgencia}
                      </span>
                    </div>
                  </div>
                </div>

                {/* TARJETA 2: Módulo Visión */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                      <i className="fa-solid fa-eye text-indigo-400"></i>
                      Diagnóstico Visión (YOLO)
                    </span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border ${
                        evalResult.vision.es_confiable
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                      }`}
                    >
                      {evalResult.vision.es_confiable ? 'Confiable ≥70%' : 'Baja Confianza'}
                    </span>
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-slate-400">Clase Visual:</span>
                      <span className="font-bold text-white truncate max-w-[170px]">
                        {evalResult.vision.clase_imagen}
                      </span>
                    </div>

                    <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-indigo-500 transition-all duration-500"
                        style={{
                          width: `${Math.min(
                            100,
                            evalResult.vision.confianza_imagen * 100
                          )}%`
                        }}
                      ></div>
                    </div>
                    <div className="text-right text-[10px] text-slate-500 mt-1 font-mono">
                      Confianza: {(evalResult.vision.confianza_imagen * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* Top-3 de Visión */}
                  <div className="pt-2 border-t border-slate-800/80 space-y-1">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                      Top-3 Clasificación:
                    </span>
                    {evalResult.vision.top3.map((item, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between text-[11px] text-slate-400"
                      >
                        <span className="truncate max-w-[160px]">{item.clase}</span>
                        <span className="font-mono text-slate-300">
                          {(item.confianza * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Visor JSON Crudo Desplegable */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <button
                  onClick={() => setShowRawJson(!showRawJson)}
                  className="w-full px-4 py-3 text-left text-xs font-semibold text-slate-300 flex items-center justify-between hover:bg-slate-800/50 transition"
                >
                  <span className="flex items-center gap-2">
                    <i className="fa-solid fa-code text-teal-400"></i>
                    Respuesta JSON Completa (Auditoría / Sustentación)
                  </span>
                  <i
                    className={`fa-solid fa-chevron-down transition-transform ${
                      showRawJson ? 'rotate-180' : ''
                    }`}
                  ></i>
                </button>

                {showRawJson && (
                  <div className="p-4 border-t border-slate-800 bg-slate-950">
                    <div className="flex justify-end mb-2">
                      <button
                        onClick={copyToClipboard}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-200 border border-slate-700 flex items-center gap-1.5 transition"
                      >
                        <i
                          className={`fa-solid ${
                            copiedJson ? 'fa-check text-emerald-400' : 'fa-copy'
                          }`}
                        ></i>
                        {copiedJson ? 'Copiado!' : 'Copiar JSON'}
                      </button>
                    </div>
                    <pre className="text-[11px] font-mono text-emerald-400/90 bg-slate-900/90 p-3 rounded-lg overflow-x-auto max-h-64 border border-slate-800">
                      {JSON.stringify(evalResult, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 px-6 py-4 text-center text-xs text-slate-500 bg-slate-950">
        Sistema de Triaje Multimodal Clínico • Prototipo Integrador Seminario III • React + FastAPI + YOLOv8 + Scikit-Learn
      </footer>
    </div>
  )
}
