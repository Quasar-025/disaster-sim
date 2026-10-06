import { useState, useEffect, useRef, useCallback } from 'react';

export function useDisasterState() {
  const [worldState, setWorldState] = useState(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState([]);
  const [agentTrails, setAgentTrails] = useState({});
  const wsRef = useRef(null);
  const prevVictimsRef = useRef({});

  useEffect(() => {
    let ws = new WebSocket('ws://localhost:8000/ws');
    wsRef.current = ws;
    
    ws.onopen = () => {
      console.log('Connected to Disaster Sim backend');
      setConnected(true);
    };
    
    ws.onclose = () => {
      console.log('Disconnected');
      setConnected(false);
      setTimeout(() => {
        setConnected(false);
      }, 2000);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setWorldState(data);
        
        // --- Derive Event Log ---
        if (data.victims) {
          const newEvents = [];
          const timeStr = `T+${data.timestep}`;
          
          Object.entries(data.victims).forEach(([vid, v]) => {
            const prevV = prevVictimsRef.current[vid];
            if (!prevV) return; // Skip initial load
            
            // Newly detected
            if (v.detected && !prevV.detected) {
              newEvents.push({
                id: `${vid}-det-${data.timestep}`,
                time: timeStr,
                type: 'detection',
                message: `Drone detected ${v.severity_name} victim at (${v.row}, ${v.col})`
              });
            }
            
            // Ambulance dispatched
            if (v.assigned_agent && !prevV.assigned_agent) {
              newEvents.push({
                id: `${vid}-disp-${data.timestep}`,
                time: timeStr,
                type: 'dispatch',
                message: `${v.assigned_agent} dispatched to victim ${vid.split('_')[1]}`
              });
            }
            
            // Rescued
            if (v.rescued && !prevV.rescued) {
              newEvents.push({
                id: `${vid}-resc-${data.timestep}`,
                time: timeStr,
                type: 'rescue',
                message: `${v.assigned_agent} rescued victim from (${v.row}, ${v.col})`
              });
            }
            
            // Transported (Drop off)
            if (v.transported && !prevV.transported) {
              newEvents.push({
                id: `${vid}-trans-${data.timestep}`,
                time: timeStr,
                type: 'transport',
                message: `Victim safely transported to hospital`
              });
            }
          });
          
          if (newEvents.length > 0) {
            setEvents(prev => [...newEvents, ...prev].slice(0, 50)); // Keep last 50 events
          }
          
          prevVictimsRef.current = data.victims;
        }

        // --- Track Agent Trails ---
        if (data.agents) {
          setAgentTrails(prev => {
            const newTrails = { ...prev };
            Object.entries(data.agents).forEach(([aid, a]) => {
              if (!newTrails[aid]) newTrails[aid] = [];
              newTrails[aid].push({ r: a.row, c: a.col });
              if (newTrails[aid].length > 30) {
                newTrails[aid].shift(); // Keep last 30 positions
              }
            });
            return newTrails;
          });
        }
        
      } catch (err) {
        console.error('Error parsing WS message', err);
      }
    };
    
    return () => {
      ws.close();
    };
  }, []);

  const setSimulationSpeed = useCallback((speed) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_speed', value: speed }));
    }
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

  return { 
    worldState, 
    connected, 
    events, 
    agentTrails, 
    setSimulationSpeed, 
    injectEarthquake, 
    injectFlood 
  };
}
