const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { test } = require('node:test');
const { runInNewContext } = require('node:vm');

const source = readFileSync(`${__dirname}/main.js`, 'utf8');
const config = { apiKey: 'fixture-key', requestPath: 'https://ocr.example/v1/chat/completions' };

function loadPlugin(fetch, tauriFetch = async () => ({
    ok: true,
    data: { choices: [{ message: { content: 'synchronous OCR' } }] },
})) {
    const recognize = runInNewContext(`${source}\nrecognize`, { window: { fetch }, TextDecoder });
    return (options = {}) => recognize('image-data', 'zh_tw', {
        config,
        utils: { tauriFetch },
        ...options,
    });
}

function sse(body) {
    const bytes = new TextEncoder().encode(body);
    let offset = 0;
    const stream = new ReadableStream({ pull(controller) {
        if (offset === bytes.length) {
            controller.close();
        } else {
            controller.enqueue(bytes.subarray(offset, ++offset));
        }
    } });
    return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
}

test('OCR streaming defaults on and reports cumulative visible text across byte-split SSE', async () => {
    const updates = [];
    const controller = new AbortController();
    const recognize = loadPlugin(async (url, options) => {
        assert.equal(url, config.requestPath);
        assert.equal(options.signal, controller.signal);
        const payload = JSON.parse(options.body);
        assert.equal(payload.stream, true);
        assert.equal(payload.messages[0].content[0].text, 'Recognize zh_tw');
        assert.equal(payload.messages[0].content[1].image_url.url, 'data:image/png;base64,image-data');
        return sse(': heartbeat\r\n\r\n' +
            'data: {"choices":[{"delta":{"reasoning_content":"internal"}}]}\r\n\r\n' +
            'data: {"choices":[\r\ndata: {"delta":{"content":"辨識"}}]}\r\n\r\n' +
            'data: {"choices":[{"delta":{"content":"文字"}}]}\n\n' +
            'data: [DONE]\n\n');
    });

    assert.equal(await recognize({
        config: { ...config, customPrompt: 'Recognize $lang' },
        setResult: (value) => updates.push(value),
        signal: controller.signal,
    }), '辨識文字');
    assert.deepEqual(updates, ['辨識', '辨識文字']);
});

test('an explicit false keeps the Tauri synchronous request available', async () => {
    const recognize = loadPlugin(() => assert.fail('Unexpected streaming request'), async (url, options) => {
        assert.equal(url, config.requestPath);
        assert.equal(options.body.payload.stream, undefined);
        return { ok: true, data: { choices: [{ message: { content: '同步辨識' } }] } };
    });

    assert.equal(await recognize({ config: { ...config, stream: false } }), '同步辨識');
});

test('an OpenAI finish reason allows clean EOF without a DONE marker', async () => {
    const recognize = loadPlugin(async () => sse(
        'data: {"choices":[{"delta":{"content":"完成"},"finish_reason":"stop"}]}\n\n'
    ));
    assert.equal(await recognize(), '完成');
});

test('unannotated EOF rejects after yielding partial visible text', async () => {
    const updates = [];
    const recognize = loadPlugin(async () => sse('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'));
    await assert.rejects(recognize({ setResult: (value) => updates.push(value) }), /ended before completion/i);
    assert.deepEqual(updates, ['partial']);
});

test('a trailing explicit error rejects even after a choice finished', async () => {
    const recognize = loadPlugin(async () => sse(
        'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"stop"}]}\n\n' +
        'event: error\ndata: {"error":{"message":"request contains fixture-key"}}\n\n'
    ));
    await assert.rejects(recognize(), (error) => /stream/i.test(error.message) && !error.message.includes('fixture-key'));
});

test('a length-limited OCR response rejects even if visible text was produced', async () => {
    const recognize = loadPlugin(async () => sse(
        'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
    ));
    await assert.rejects(recognize(), /length|limit/i);
});

test('a finished stream containing only internal reasoning does not return an empty OCR result', async () => {
    const recognize = loadPlugin(async () => sse(
        'data: {"choices":[{"delta":{"reasoning_content":"internal"},"finish_reason":"stop"}]}\n\n'
    ));
    await assert.rejects(recognize(), /no visible text/i);
});

test('native Gemini uses its SSE endpoint and excludes thought parts from the OCR text', async () => {
    const updates = [];
    const recognize = loadPlugin(async (url, options) => {
        assert.equal(url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse');
        assert.equal(options.headers['x-goog-api-key'], config.apiKey);
        const payload = JSON.parse(options.body);
        assert.equal(payload.contents[0].parts[1].inlineData.data, 'image-data');
        return sse('data: {"candidates":[{"index":0,"content":{"parts":[{"text":"thinking","thought":true},{"text":"辨識"}]}}]}\n\n' +
            'data: {"candidates":[{"index":0,"content":{"parts":[{"text":"完成"}]},"finishReason":"STOP"}]}\n\n');
    });

    assert.equal(await recognize({
        config: { ...config, requestPath: 'https://generativelanguage.googleapis.com', model: 'gemini-2.5-flash' },
        setResult: (value) => updates.push(value),
    }), '辨識完成');
    assert.deepEqual(updates, ['辨識', '辨識完成']);
});

test('stream HTTP failures expose the status without leaking provider response text', async () => {
    const recognize = loadPlugin(async () => new Response('request contains fixture-key', { status: 524 }));
    await assert.rejects(recognize(), (error) => /524/.test(error.message) && !error.message.includes('fixture-key'));
});

test('aborting the request rejects without falling back to a synchronous request', async () => {
    const controller = new AbortController();
    const recognize = loadPlugin(async (url, { signal }) => new Promise((resolve, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), { once: true });
    }));
    const result = recognize({ signal: controller.signal });
    controller.abort();
    await assert.rejects(result, { name: 'AbortError' });
});

test('plugin settings expose an enabled-by-default streaming switch', () => {
    const { needs } = JSON.parse(readFileSync(`${__dirname}/info.json`, 'utf8'));
    const streaming = needs.find(({ key }) => key === 'stream');
    assert.equal(streaming?.type, 'switch');
    assert.equal(streaming?.default, true);
});
