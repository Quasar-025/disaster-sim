import React, { useRef, useEffect } from 'react';

// Terrain enum mapping from Python:
// 0: EMPTY, 1: ROAD, 2: BUILDING, 3: PARK, 4: WATER
// 5: HOSPITAL, 6: CHARGING_STATION, 7: DEBRIS
// 8: BUILDING_DAMAGED, 9: FIRE

const TERRAIN_COLORS = {
  0: '#1a1a24', // Empty
  1: '#3a3a44', // Road
  2: '#4a4a54', // Building
  3: '#8a2b2b', // Damaged Building
  4: '#8b5a2b', // Debris
  5: '#1a5b7c', // Water
  6: '#ff4500', // Fire
  7: '#ffffff', // Hospital
  8: '#ffff00', // Charging Station
  9: '#2d4c2b', // Park
};

export default function StrategicMap({ worldState, width = 600, height = 600, onCellClick }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!worldState || !worldState.grid) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // grid is a 2D array: list of lists
    const gridHeight = worldState.grid.length;
    const gridWidth = worldState.grid[0].length;
    const cellW = width / gridWidth;
    const cellH = height / gridHeight;

    // Clear
    ctx.clearRect(0, 0, width, height);

    for (let r = 0; r < gridHeight; r++) {
      for (let c = 0; c < gridWidth; c++) {
        const val = worldState.grid[r][c];
        ctx.fillStyle = TERRAIN_COLORS[val] || '#000';
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        
        // Overlay traffic GNN predictions
        const key = `${r},${c}`;
        if (worldState.predictive_traffic && worldState.predictive_traffic[key]) {
           const prob = worldState.predictive_traffic[key];
           if (prob > 0.2) {
             ctx.fillStyle = `rgba(255, 0, 0, ${prob * 0.8})`;
             ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
           }
        }
      }
    }
    
    // Draw Agents
    if (worldState.agents) {
      Object.values(worldState.agents).forEach(agent => {
        const { row, col, type, status } = agent;
        ctx.beginPath();
        ctx.arc(col * cellW + cellW/2, row * cellH + cellH/2, cellW/2, 0, 2 * Math.PI);
        if (type === 'ambulance') ctx.fillStyle = '#ff0000';
        else if (type === 'drone') ctx.fillStyle = '#00ffff';
        else if (type === 'traffic_light') {
           ctx.fillStyle = status === 'green' || status === 'preempted_green' ? '#00ff00' : '#ff0000';
        }
        else ctx.fillStyle = '#ff00ff';
        ctx.fill();
      });
    }

    // Draw Victims
    if (worldState.victims) {
      Object.values(worldState.victims).forEach(v => {
        if (!v.rescued) {
          ctx.fillStyle = '#ff8800';
          ctx.fillRect(v.col * cellW + cellW*0.25, v.row * cellH + cellH*0.25, cellW*0.5, cellH*0.5);
        }
      });
    }
    
  }, [worldState, width, height]);

  const handleClick = (e) => {
    if (!worldState || !onCellClick) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const gridHeight = worldState.grid.length;
    const gridWidth = worldState.grid[0].length;
    const cellW = width / gridWidth;
    const cellH = height / gridHeight;
    
    const c = Math.floor(x / cellW);
    const r = Math.floor(y / cellH);
    
    onCellClick(r, c);
  };

  return (
    <div style={{ border: '1px solid #333', borderRadius: '8px', overflow: 'hidden', cursor: 'crosshair', background: '#000' }}>
      <canvas 
        ref={canvasRef} 
        width={width} 
        height={height} 
        onClick={handleClick}
      />
    </div>
  );
}
