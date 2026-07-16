import { useState, useEffect } from 'react';

export function useDisasterState() {
  const [worldState, setWorldState] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('Connected to Disaster Sim backend');
      setConnected(true);
    };
    
    ws.onclose = () => {
      console.log('Disconnected');
      setConnected(false);
      // Auto-reconnect after 2s
      setTimeout(() => {
        setConnected(false); // force re-render if needed
      }, 2000);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setWorldState(data);
      } catch (err) {
        console.error('Error parsing WS message', err);
      }
    };
    
    return () => {
      ws.close();
    };
  }, []);

  // REST API hooks for disaster injection
  const injectEarthquake = async (r, c, magnitude) => {
    try {
      await fetch('http://localhost:8000/api/disaster/earthquake', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ r, c, magnitude })
      });
    } catch(e) { console.error(e) }
  };

  const injectFlood = async (r, c, volume) => {
    try {
      await fetch('http://localhost:8000/api/disaster/flood', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ r, c, volume })
      });
    } catch(e) { console.error(e) }
  };

  return { worldState, connected, injectEarthquake, injectFlood };
}
