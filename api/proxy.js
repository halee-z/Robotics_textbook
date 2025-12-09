module.exports = async (req, res) => {
  try {
    // Replace with your real target API URL
    const targetUrl = process.env.TARGET_API_URL || 'https://example-remote-api.com/endpoint';

    const method = req.method || 'GET';

    // Build headers for upstream request, include server-side API key from env
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.MY_API_KEY}`,
    };

    let body;
    if (method !== 'GET' && req.body) {
      body = typeof req.body === 'string' ? req.body : JSON.stringify(req.body);
    }

    const upstream = await fetch(targetUrl, { method, headers, body });
    const text = await upstream.text();

    // Forward status and content-type
    res.status(upstream.status);
    const ct = upstream.headers.get('content-type') || 'application/json';
    res.setHeader('content-type', ct);
    return res.send(text);
  } catch (err) {
    console.error('Proxy error', err);
    res.status(500).json({ error: 'Proxy error', details: err?.message || String(err) });
  }
};
