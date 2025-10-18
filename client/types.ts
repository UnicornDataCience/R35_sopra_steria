// Type definitions for Patient-IA Dashboard

export interface PatientData {
    id: string;
    age: number;
    gender: 'M' | 'F';
    diagnosis: string;
    admissionDate: Date;
    events: MedicalEvent[];
}

export interface MedicalEvent {
    id: string;
    patientId: string;
    eventType: 'diagnosis' | 'treatment' | 'lab_result' | 'procedure';
    description: string;
    timestamp: Date;
    severity?: 'low' | 'medium' | 'high' | 'critical';
}

export interface DatasetStats {
    totalPatients: number;
    averageAge: number;
    totalEvents: number;
    totalRecords: number;
}

export interface ChartData {
    temporal: {
        labels: string[];
        data: number[];
    };
    diagnostics: {
        labels: string[];
        data: number[];
        colors: string[];
    };
}

export interface ChatMessage {
    id: string;
    content: string;
    sender: 'user' | 'assistant';
    timestamp: Date;
    type?: 'text' | 'data' | 'analysis';
}

export interface SidebarSection {
    id: string;
    title: string;
    items: SidebarItem[];
}

export interface SidebarItem {
    id: string;
    label: string;
    icon: string;
    action: string;
    active?: boolean;
}

export interface DashboardConfig {
    theme: 'light' | 'dark';
    language: 'es' | 'en';
    refreshInterval: number;
    enableNotifications: boolean;
}

export interface FileUploadResult {
    filename: string;
    size: number;
    type: string;
    columns: string[];
    rows: number;
    preview: any[];
}

export interface AnalysisResult {
    summary: DatasetStats;
    quality: QualityMetrics;
    recommendations: string[];
    visualizations: ChartData;
}

export interface QualityMetrics {
    completeness: number;
    consistency: number;
    accuracy: number;
    timeliness: number;
    validity: number;
}

export interface SyntheticDataConfig {
    method: 'ctgan' | 'tvae' | 'sdv';
    numberOfRecords: number;
    privacyLevel: 'low' | 'medium' | 'high';
    preserveRelations: boolean;
}

export interface SimulationConfig {
    patientProfile: PatientProfile;
    timespan: number;
    eventFrequency: 'low' | 'medium' | 'high';
    includeComplications: boolean;
}

export interface PatientProfile {
    age: number;
    gender: 'M' | 'F';
    conditions: string[];
    riskFactors: string[];
}

// Event types for the dashboard
export type DashboardEvent = 
    | 'file_uploaded'
    | 'analysis_started'
    | 'analysis_completed'
    | 'generation_started'
    | 'generation_completed'
    | 'chat_message_sent'
    | 'navigation_changed';

// API response types
export interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
    message?: string;
}

export interface UploadResponse extends ApiResponse<FileUploadResult> {}
export interface AnalysisResponse extends ApiResponse<AnalysisResult> {}
export interface GenerationResponse extends ApiResponse<{url: string; size: number}> {}

// Dashboard state interface
export interface DashboardState {
    currentSection: string;
    loadedDataset?: FileUploadResult;
    lastAnalysis?: AnalysisResult;
    chatHistory: ChatMessage[];
    isLoading: boolean;
    notifications: Notification[];
}

export interface Notification {
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message: string;
    timestamp: Date;
    autoClose?: boolean;
}