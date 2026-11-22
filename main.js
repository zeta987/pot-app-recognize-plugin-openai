async function* readSse(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let line = '';
    let skipLineFeed = false;
    let data = [];
    let event = '';
    let done = false;

    try {
        while (!done) {
            const chunk = await reader.read();
            done = chunk.done;
            const decoded = decoder.decode(chunk.value, { stream: !done });
            if (!decoded) {
                continue;
            }

            const text = skipLineFeed && decoded.startsWith('\n') ? decoded.slice(1) : decoded;
            skipLineFeed = decoded.endsWith('\r');
            const lines = text.split(/\r\n|\r|\n/);
            lines[0] = line + lines[0];
            line = lines.pop();

            for (const completeLine of lines) {
                if (completeLine === '') {
                    if (data.length || event === 'error') {
                        yield { event, data: data.join('\n') };
                    }
                    data = [];
                    event = '';
                    continue;
                }

                const separator = completeLine.indexOf(':');
                const field = separator === -1 ? completeLine : completeLine.slice(0, separator);
                const value = separator === -1 ? '' : completeLine.slice(separator + 1).replace(/^ /, '');
                if (field === 'data') {
                    data.push(value);
                } else if (field === 'event') {
                    event = value;
                }
            }
        }
    } finally {
        if (!done) {
            // Reader cleanup must not replace the stream result or error.
            reader.cancel().catch(() => {});
        }
        reader.releaseLock();
    }
}

async function recognizeStream(requestPath, headers, body, isGoogleAPI, setResult, signal) {
    const response = await window.fetch(requestPath, {
        method: 'POST',
        headers: { ...headers, Accept: 'text/event-stream' },
        body: JSON.stringify(body),
        signal,
    });
    if (!response.ok) {
        response.body?.cancel().catch(() => {});
        throw new Error(`Request failed with status ${response.status}`);
    }
    if (!response.body || !/^text\/event-stream(?:\s*;|$)/i.test(response.headers.get('Content-Type') ?? '')) {
        response.body?.cancel().catch(() => {});
        throw new Error('OCR API did not return an event stream.');
    }

    let content = '';
    let finished = false;
    for await (const { event, data } of readSse(response.body)) {
        if (event === 'error') {
            throw new Error('OCR API sent a stream error.');
        }
        if (!data.trim()) {
            continue;
        }
        if (!isGoogleAPI && data.trim() === '[DONE]') {
            finished = true;
            break;
        }

        let result;
        try {
            result = JSON.parse(data);
        } catch {
            throw new Error('OCR API sent invalid stream data.');
        }
        if (result?.error) {
            throw new Error('OCR API sent a stream error.');
        }

        const alternatives = isGoogleAPI ? result?.candidates : result?.choices;
        const choice = alternatives?.find((item, index) => (item?.index ?? index) === 0);
        const finishReason = isGoogleAPI ? choice?.finishReason : choice?.finish_reason;
        const blockReason = result?.promptFeedback?.blockReason;
        if (finishReason === 'length' || finishReason === 'MAX_TOKENS') {
            throw new Error('OCR output reached the token limit.');
        }
        if ((blockReason && blockReason !== 'BLOCK_REASON_UNSPECIFIED') || choice?.delta?.refusal ||
            ['content_filter', 'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII',
                'IMAGE_SAFETY', 'IMAGE_PROHIBITED_CONTENT'].includes(finishReason)) {
            throw new Error('OCR output was blocked by the provider.');
        }
        if (finishReason && finishReason !== 'FINISH_REASON_UNSPECIFIED') {
            if (finishReason !== (isGoogleAPI ? 'STOP' : 'stop')) {
                throw new Error('OCR stream stopped before successful recognition.');
            }
            finished = true;
        }

        const delta = isGoogleAPI
            ? choice?.content?.parts?.filter((part) => !part.thought && typeof part.text === 'string')
                .map((part) => part.text).join('')
            : choice?.delta?.content;
        if (typeof delta === 'string' && delta) {
            content += delta;
            setResult?.(content);
        }
    }

    if (!finished) {
        throw new Error('OCR stream ended before completion.');
    }
    if (!content.trim()) {
        throw new Error('OCR stream returned no visible text.');
    }
    return content;
}

async function recognize(base64, lang, options) {
    const { config, utils, setResult, signal } = options;
    const { tauriFetch: fetch } = utils;
    let { model = "gpt-4o", apiKey, requestPath, customPrompt } = config;
    const stream = config.stream !== false;

    // 檢查是否是 Google API
    const isGoogleAPI = requestPath?.includes('generativelanguage.googleapis.com');
    
    if (isGoogleAPI) {
        // Google Gemini API 格式
        if (!model || model === "gpt-4o") {
            model = "gemini-1.5-flash";
        }
        
        const headers = {
            'Content-Type': 'application/json'
        }

        // 使用正確的 Google API endpoint
        requestPath = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

        const body = {
            contents: [{
                role: "user",
                parts: [
                    {
                        text: customPrompt || "Just recognize the text in the image. Do not offer unnecessary explanations."
                    },
                    {
                        inlineData: {
                            mimeType: "image/png",
                            data: base64
                        }
                    }
                ]
            }]
        };

        if (stream) {
            const streamPath = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;
            return recognizeStream(streamPath, { ...headers, 'x-goog-api-key': apiKey }, body, true, setResult, signal);
        }

        // 添加 API key 到 URL
        requestPath += `?key=${apiKey}`;

        let res = await fetch(requestPath, {
            method: 'POST',
            url: requestPath,
            headers: headers,
            body: {
                type: "Json",
                payload: body
            },
            responseType: 1
        });

        // 處理 Google API 的返回格式
        if (res.ok) {
            let result = res.data;
            // 檢查返回格式並提供詳細錯誤信息
            if (!result) {
                throw `Empty response from Google API`;
            }
            if (!result.candidates) {
                throw `No candidates in response: ${JSON.stringify(result)}`;
            }
            if (!result.candidates[0]) {
                throw `Empty candidates array: ${JSON.stringify(result)}`;
            }
            if (!result.candidates[0].content) {
                throw `No content in candidate: ${JSON.stringify(result.candidates[0])}`;
            }
            if (!result.candidates[0].content.parts) {
                throw `No parts in content: ${JSON.stringify(result.candidates[0].content)}`;
            }
            if (!result.candidates[0].content.parts[0]) {
                throw `Empty parts array: ${JSON.stringify(result.candidates[0].content.parts)}`;
            }
            if (!result.candidates[0].content.parts[0].text) {
                throw `No text in part: ${JSON.stringify(result.candidates[0].content.parts[0])}`;
            }
            return result.candidates[0].content.parts[0].text;
        } else {
            throw `Request failed with status ${res.status}: ${JSON.stringify(res.data)}`;
        }

    } else {
        // OpenAI API 處理邏輯
        if (!requestPath) {
            requestPath = "https://api.openai.com";
        }
        if (!/https?:\/\/.+/.test(requestPath)) {
            requestPath = `https://${requestPath}`;
        }
        if (requestPath.endsWith('/')) {
            requestPath = requestPath.slice(0, -1);
        }
        if (!requestPath.endsWith('/chat/completions')) {
            requestPath += '/v1/chat/completions';
        }
        if (!customPrompt) {
            customPrompt = "Just recognize the text in the image. Do not offer unnecessary explanations.";
        }else{
            customPrompt = customPrompt.replaceAll("$lang", lang);
        }

        const headers = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        }

        const body = {
            model,
            messages: [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": customPrompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": `data:image/png;base64,${base64}`,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        };

        if (stream) {
            return recognizeStream(requestPath, headers, { ...body, stream: true }, false, setResult, signal);
        }

        let res = await fetch(requestPath, {
            method: 'POST',
            url: requestPath,
            headers: headers,
            body: {
                type: "Json",
                payload: body
            },
            responseType: 1
        });

        /* Debug 相關代碼
        const debugBody = { ...body };
        if (debugBody.messages?.[0]?.content) {
            debugBody.messages[0].content = debugBody.messages[0].content.map(item => {
                if (item.type === 'image_url') {
                    return { 
                        type: 'image_url', 
                        image_url: { 
                            url: '***[BASE64]***',
                            detail: item.image_url.detail 
                        } 
                    };
                }
                return item;
            });
        }
        */

        if (res.ok) {
            let result = res.data;
            if (!result || !result.choices || !result.choices[0]) {
                throw `Invalid API Response: ${JSON.stringify(result)}`;
            }
            
            const choice = result.choices[0];
            let content = '';
            if (choice.message && choice.message.content) {
                content = choice.message.content;
            } else if (choice.content) {
                content = choice.content;
            } else {
                content = JSON.stringify(choice);
            }

            /* Debug 相關代碼
            return `🔍 Debug Info:
Request URL: ${requestPath}
Request Headers: ${JSON.stringify(headers, null, 2)}
Request Body: ${JSON.stringify(debugBody, null, 2)}

📝 Response:
${content}`;
            */
            return content;
        } else {
            /* Debug 相關代碼
            throw `❌ Http Request Error
URL: ${requestPath}
Status: ${res.status}
Headers: ${JSON.stringify(headers, null, 2)}
Request Body: ${JSON.stringify(debugBody, null, 2)}
Response: ${JSON.stringify(res.data, null, 2)}`;
            */
            throw `Request failed with status ${res.status}`;
        }
    }
}
