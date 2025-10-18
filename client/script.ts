document.addEventListener('DOMContentLoaded', () => {
    // Element selectors
    const uploadBtn = document.getElementById('upload-btn');
    const uploadSidebarBtn = document.getElementById('upload-sidebar-btn');
    const fileUploadInput = document.getElementById('file-upload') as HTMLInputElement;
    const chatInput = document.getElementById('chat-input') as HTMLInputElement;
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const datasetList = document.getElementById('dataset-list');

    const API_BASE_URL = 'http://127.0.0.1:8000'; // Asume que la API corre en el puerto 8000

    // --- Dataset Upload Logic ---
    const triggerFileUpload = () => {
        fileUploadInput?.click();
    };

    const handleFileUpload = async (event: Event) => {
        const target = event.target as HTMLInputElement;
        const file = target.files?.[0];
        if (!file) return;

        addMessageToChat('user', `Subiendo archivo: ${file.name}`);
        showTypingIndicator();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/api/datasets/upload`, {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();
            removeTypingIndicator();

            if (response.ok && result.success) {
                const datasetInfo = result.data.dataset_info;
                addMessageToChat('assistant', `Archivo '${datasetInfo.filename}' subido con éxito. ID: ${datasetInfo.id}`);
                fetchAndDisplayDatasets(); // Refresh the list
            } else {
                throw new Error(result.message || 'Error al subir el archivo.');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            removeTypingIndicator();
            addMessageToChat('assistant', `Error al subir el archivo: ${error.message}`);
        }
    };

    uploadBtn?.addEventListener('click', triggerFileUpload);
    uploadSidebarBtn?.addEventListener('click', triggerFileUpload);
    fileUploadInput?.addEventListener('change', handleFileUpload);

    // --- Dataset List Logic ---
    const fetchAndDisplayDatasets = async () => {
        if (!datasetList) return;
        try {
            const response = await fetch(`${API_BASE_URL}/api/datasets`);
            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.message || 'No se pudieron cargar los datasets.');
            }

            datasetList.innerHTML = ''; // Clear current list
            const datasets = result.data.datasets;
            
            if (datasets.length === 0) {
                datasetList.innerHTML = '<li class="dataset-list-item-empty">No hay datasets cargados.</li>';
            } else {
                datasets.forEach((dataset: any) => {
                    const listItem = document.createElement('li');
                    listItem.className = 'dataset-list-item';
                    listItem.innerHTML = `
                        <i class="fas fa-file-csv"></i>
                        <div class="dataset-item-info">
                            <span class="dataset-name">${dataset.filename}</span>
                            <span class="dataset-id">ID: ${dataset.id}</span>
                        </div>
                    `;
                    listItem.dataset.id = dataset.id;
                    datasetList.appendChild(listItem);
                });
            }

        } catch (error) {
            console.error('Error fetching datasets:', error);
            datasetList.innerHTML = '<li class="dataset-list-item-empty">Error al cargar.</li>';
        }
    };

    // --- Chat Logic ---
    const sendMessage = async () => {
        const message = chatInput.value.trim();
        if (!message) return;

        addMessageToChat('user', message);
        chatInput.value = '';
        showTypingIndicator();

        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/message`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            const result = await response.json();
            removeTypingIndicator();

            if (response.ok && result.success) {
                const botResponse = result.data.response || 'No he podido procesar tu petición.';
                addMessageToChat('assistant', botResponse);
            } else {
                throw new Error(result.message || 'Error en la respuesta de la API.');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            removeTypingIndicator();
            addMessageToChat('assistant', `Lo siento, hubo un error: ${error.message}`);
        }
    };

    sendBtn?.addEventListener('click', sendMessage);
    chatInput?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // --- Helper Functions ---
    const addMessageToChat = (sender: 'user' | 'assistant', message: string) => {
        if (!chatContainer) return;

        const messageWrapper = document.createElement('div');
        messageWrapper.className = `chat-message ${sender}`;

        const icon = document.createElement('div');
        icon.className = 'message-icon';
        const iconAwesome = document.createElement('i');
        iconAwesome.className = `fas ${sender === 'user' ? 'fa-user' : 'fa-robot'}`;
        icon.appendChild(iconAwesome);

        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = message;

        messageWrapper.appendChild(icon);
        messageWrapper.appendChild(content);
        chatContainer.appendChild(messageWrapper);

        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const showTypingIndicator = () => {
        if (!chatContainer) return;
        const typingIndicator = document.createElement('div');
        typingIndicator.id = 'typing-indicator';
        typingIndicator.className = 'chat-message assistant';
        typingIndicator.innerHTML = `
            <div class="message-icon">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatContainer.appendChild(typingIndicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const removeTypingIndicator = () => {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    };

    // Initial Load
    fetchAndDisplayDatasets();
});
