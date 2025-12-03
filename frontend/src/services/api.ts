import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const mobileChat = async (message: string, sessionId?: string) => {
    const response = await api.post('/mobile/chat', { message, session_id: sessionId });
    return response.data;
};

export const kioskScan = async (sessionId: string) => {
    const response = await api.post('/kiosk/scan', { session_id: sessionId });
    return response.data;
};
