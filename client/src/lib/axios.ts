import axios from "axios";

export const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api",
  timeout: 120000,
  headers: {
    "Accept": "application/json",
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.data) {
      const message =
        error.response.data.detail ||
        error.response.data.message ||
        `HTTP Error ${error.response.status}`;
      return Promise.reject(new Error(message));
    }
    return Promise.reject(error);
  }
);
