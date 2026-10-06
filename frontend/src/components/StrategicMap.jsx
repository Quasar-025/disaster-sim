import React, { useRef, useEffect, useState } from 'react';

const TERRAIN_COLORS = {
  0: '#111827', // Empty
  1: '#374151', // Road
  2: '#4b5563', // Building
  3: '#b91c1c', // Damaged Building
  4: '#92400e', // Debris
  5: '#0369a1', // Water
  6: '#ea580c', // Fire
  7: '#ffffff', // Hospital
  8: '#facc15', // Charging Station
  9: '#166534', // Park
};

const SEVERITY_COLORS = {
  0: '#ef4444', // Critical (Red)
  1: '#f97316', // Serious (Orange)
  2: '#eab308', // Stable (Yellow)
};

export default function StrategicMap({ worldState, agentTrails, onCellClick }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Auto-resize canvas to fit parent
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (let entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!worldState || !worldState.grid || dimensions.width === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const gridHeight = worldState.grid.length;
    const gridWidth = worldState.grid[0].length;
    const cellW = dimensions.width / gridWidth;
    const cellH = dimensions.height / gridHeight;

    // Clear
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    // 1. Draw Base Terrain
    for (let r = 0; r < gridHeight; r++) {
      for (let c = 0; c < gridWidth; c++) {
        const val = worldState.grid[r][c];
        ctx.fillStyle = TERRAIN_COLORS[val] || '#000';
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }
    }
    
    // 2. Draw Fog of War (Dark overlay on unexplored)
    if (worldState.explored) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      for (let r = 0; r < gridHeight; r++) {
        for (let c = 0; c < gridWidth; c++) {
          if (!worldState.explored[r][c]) {
            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
          }
        }
      }
    }

    // 3. Draw Hospitals (Base Stations)
    if (worldState.hospitals) {
      Object.values(worldState.hospitals).forEach(h => {
        ctx.fillStyle = '#fff';
        ctx.fillRect(h.col * cellW, h.row * cellH, cellW, cellH);
        
        // Red cross
        ctx.fillStyle = '#ef4444';
        const cx = h.col * cellW + cellW/2;
        const cy = h.row * cellH + cellH/2;
        const cw = cellW * 0.6;
        const cw2 = cellW * 0.2;
        ctx.fillRect(cx - cw/2, cy - cw2/2, cw, cw2);
        ctx.fillRect(cx - cw2/2, cy - cw/2, cw2, cw);
      });
    }

    // 4. Draw Victims
    if (worldState.victims) {
      Object.values(worldState.victims).forEach(v => {
        if (!v.rescued) {
          const cx = v.col * cellW + cellW/2;
          const cy = v.row * cellH + cellH/2;
          const radius = cellW * 0.6;
          
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
          ctx.fillStyle = SEVERITY_COLORS[v.severity] || '#fff';
          ctx.fill();
          
          // Pulse if detected but not assigned
          if (v.detected && !v.assigned_agent) {
             ctx.beginPath();
             ctx.arc(cx, cy, radius * 1.5 + Math.sin(Date.now() / 100) * radius * 0.5, 0, 2 * Math.PI);
             ctx.strokeStyle = SEVERITY_COLORS[v.severity];
             ctx.lineWidth = 1;
             ctx.stroke();
          }
        }
      });
    }

    // 5. Draw Agent Trails & Paths
    if (worldState.agents) {
      Object.entries(worldState.agents).forEach(([aid, agent]) => {
        // Draw historic trail
        if (agentTrails && agentTrails[aid] && agentTrails[aid].length > 1) {
          ctx.beginPath();
          const trail = agentTrails[aid];
          ctx.moveTo(trail[0].c * cellW + cellW/2, trail[0].r * cellH + cellH/2);
          for (let i = 1; i < trail.length; i++) {
            ctx.lineTo(trail[i].c * cellW + cellW/2, trail[i].r * cellH + cellH/2);
          }
          ctx.strokeStyle = agent.type === 'drone' ? 'rgba(6, 182, 212, 0.4)' : 
                            agent.type === 'ambulance' ? 'rgba(239, 68, 68, 0.4)' : 'rgba(255, 255, 255, 0.2)';
          ctx.lineWidth = cellW * 0.4;
          ctx.stroke();
        }

        // Draw planned A* path for ambulances
        if (agent.path && agent.path.length > 0) {
           ctx.beginPath();
           ctx.moveTo(agent.col * cellW + cellW/2, agent.row * cellH + cellH/2);
           agent.path.forEach(([pr, pc]) => {
              ctx.lineTo(pc * cellW + cellW/2, pr * cellH + cellH/2);
           });
           ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
           ctx.setLineDash([2, 2]);
           ctx.lineWidth = Math.max(1, cellW * 0.2);
           ctx.stroke();
           ctx.setLineDash([]);
        }
      });
      
      // 6. Draw Agents on top
      Object.values(worldState.agents).forEach(agent => {
        const { row, col, type, carrying_victim } = agent;
        
        if (type === 'traffic_light') return; // Hide traffic lights to prevent visual clutter
        
        const cx = col * cellW + cellW/2;
        const cy = row * cellH + cellH/2;
        
        ctx.beginPath();
        ctx.arc(cx, cy, cellW, 0, 2 * Math.PI);
        
        if (type === 'ambulance') {
          ctx.fillStyle = '#ef4444'; // Red
          ctx.fill();
          // White cross inside ambulance
          ctx.fillStyle = '#fff';
          const cw = cellW * 1.2;
          const cw2 = cellW * 0.4;
          ctx.fillRect(cx - cw/2, cy - cw2/2, cw, cw2);
          ctx.fillRect(cx - cw2/2, cy - cw/2, cw2, cw);
        } else if (type === 'drone') {
          ctx.fillStyle = '#06b6d4'; // Cyan
          ctx.fill();
        } else if (type === 'ground_robot') {
          ctx.fillStyle = '#10b981'; // Green
          ctx.fill();
        } else {
          ctx.fillStyle = '#ff00ff';
          ctx.fill();
        }
        
        // Glow if carrying victim
        if (carrying_victim) {
          ctx.beginPath();
          ctx.arc(cx, cy, cellW * 2, 0, 2 * Math.PI);
          ctx.strokeStyle = '#facc15'; // Yellow glow
          ctx.lineWidth = cellW * 0.5;
          ctx.stroke();
        }
      });
    }

  }, [worldState, dimensions, agentTrails]);

  const handleClick = (e) => {
    if (!worldState || !onCellClick || dimensions.width === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const gridHeight = worldState.grid.length;
    const gridWidth = worldState.grid[0].length;
    const cellW = dimensions.width / gridWidth;
    const cellH = dimensions.height / gridHeight;
    
    const c = Math.floor(x / cellW);
    const r = Math.floor(y / cellH);
    
    onCellClick(r, c);
  };

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', cursor: 'crosshair', background: '#000' }}>
      <canvas 
        ref={canvasRef} 
        width={dimensions.width} 
        height={dimensions.height} 
        onClick={handleClick}
        style={{ display: 'block' }}
      />
    </div>
  );
}
