import React, { useEffect, useRef, useState } from 'react';

const TERRAIN_COLORS = {
  0: '#183038', // Open ground
  1: '#263542', // Road
  2: '#526475', // Building
  3: '#854a40', // Damaged building
  4: '#6d5549', // Debris
  5: '#147faa', // Water
  6: '#4b2b27', // Fire base
  7: '#e8f0f5', // Hospital
  8: '#159b9f', // Charging station
  9: '#3d7955', // Park
};

const SEVERITY_COLORS = {
  critical: '#ff5869',
  serious: '#ffad4e',
  stable: '#f7db75',
};

function noise(row, col, salt = 0) {
  const value = Math.sin((row * 127.1) + (col * 311.7) + (salt * 19.19)) * 43758.5453;
  return value - Math.floor(value);
}

export default function StrategicMap({ worldState, agentTrails, selectedCell, onCellClick }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!worldState?.grid || dimensions.width === 0 || dimensions.height === 0) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(dimensions.width * pixelRatio);
    canvas.height = Math.floor(dimensions.height * pixelRatio);
    canvas.style.width = `${dimensions.width}px`;
    canvas.style.height = `${dimensions.height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    const gridHeight = worldState.grid.length;
    const gridWidth = worldState.grid[0].length;
    const cellW = dimensions.width / gridWidth;
    const cellH = dimensions.height / gridHeight;
    const cellSize = Math.min(cellW, cellH);

    context.clearRect(0, 0, dimensions.width, dimensions.height);
    drawTerrain(context, worldState.grid, cellW, cellH);
    drawFogOfWar(context, worldState.explored, gridHeight, gridWidth, cellW, cellH);
    drawHospitals(context, worldState.hospitals, cellW, cellH);
    drawVictims(context, worldState.victims, cellW, cellH, cellSize);
    drawAgentPaths(context, worldState.agents, agentTrails, cellW, cellH);
    drawAgents(context, worldState.agents, cellW, cellH, cellSize);
    drawSelectedCell(context, selectedCell, cellW, cellH);
  }, [agentTrails, dimensions, selectedCell, worldState]);

  const handleClick = (event) => {
    if (!worldState?.grid || !onCellClick || dimensions.width === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const gridHeight = worldState.grid.length;
    const gridWidth = worldState.grid[0].length;
    const col = Math.max(0, Math.min(gridWidth - 1, Math.floor(x / (dimensions.width / gridWidth))));
    const row = Math.max(0, Math.min(gridHeight - 1, Math.floor(y / (dimensions.height / gridHeight))));
    onCellClick(row, col);
  };

  return (
    <div ref={containerRef} className="strategic-map" aria-label="Two dimensional strategic disaster map">
      <canvas ref={canvasRef} onClick={handleClick} />
      <div className="map-scale" aria-hidden="true"><span /> 10 cells</div>
      <div className="strategic-key" aria-hidden="true">
        <span><i className="key-road" /> Roads</span>
        <span><i className="key-park" /> Safe zones</span>
        <span><i className="key-water" /> Flooding</span>
      </div>
    </div>
  );
}

function drawTerrain(context, grid, cellW, cellH) {
  for (let row = 0; row < grid.length; row += 1) {
    for (let col = 0; col < grid[row].length; col += 1) {
      const type = grid[row][col];
      const x = col * cellW;
      const y = row * cellH;
      context.fillStyle = TERRAIN_COLORS[type] || '#122029';
      context.fillRect(x, y, cellW + 0.3, cellH + 0.3);

      // The small details make a procedural grid read as a city when viewed
      // at dashboard scale, without requiring image assets.
      if (type === 1) {
        context.fillStyle = 'rgba(192, 214, 221, 0.16)';
        if ((row + col) % 3 === 0) context.fillRect(x + cellW * 0.42, y + cellH * 0.08, Math.max(0.65, cellW * 0.16), cellH * 0.34);
      } else if (type === 2) {
        context.fillStyle = 'rgba(205, 233, 247, 0.12)';
        context.fillRect(x + cellW * 0.14, y + cellH * 0.14, Math.max(0.5, cellW * 0.18), Math.max(0.5, cellH * 0.18));
        if (noise(row, col, 4) > 0.48) context.fillRect(x + cellW * 0.62, y + cellH * 0.14, Math.max(0.45, cellW * 0.14), Math.max(0.45, cellH * 0.18));
      } else if (type === 3) {
        context.strokeStyle = 'rgba(29, 15, 14, 0.55)';
        context.lineWidth = Math.max(0.5, cellW * 0.08);
        context.beginPath();
        context.moveTo(x + cellW * 0.2, y + cellH * 0.16);
        context.lineTo(x + cellW * 0.68, y + cellH * 0.7);
        context.lineTo(x + cellW * 0.86, y + cellH * 0.52);
        context.stroke();
      } else if (type === 4) {
        context.fillStyle = 'rgba(222, 169, 121, 0.35)';
        context.fillRect(x + cellW * 0.18, y + cellH * 0.31, Math.max(0.6, cellW * 0.22), Math.max(0.6, cellH * 0.18));
        context.fillRect(x + cellW * 0.58, y + cellH * 0.51, Math.max(0.5, cellW * 0.16), Math.max(0.5, cellH * 0.13));
      } else if (type === 5 && row % 2 === 0) {
        context.strokeStyle = 'rgba(207, 246, 255, 0.26)';
        context.lineWidth = Math.max(0.45, cellW * 0.07);
        context.beginPath();
        context.moveTo(x + cellW * 0.16, y + cellH * 0.58);
        context.quadraticCurveTo(x + cellW * 0.48, y + cellH * 0.34, x + cellW * 0.82, y + cellH * 0.58);
        context.stroke();
      } else if (type === 6) {
        drawFlame(context, x + cellW / 2, y + cellH * 0.65, Math.max(1.5, cellW * 0.42));
      } else if (type === 8) {
        context.fillStyle = '#a4fff3';
        context.fillRect(x + cellW * 0.42, y + cellH * 0.14, Math.max(1, cellW * 0.16), cellH * 0.72);
      } else if (type === 9 && noise(row, col, 8) > 0.55) {
        context.fillStyle = '#6eaf70';
        context.beginPath();
        context.arc(x + cellW * 0.54, y + cellH * 0.46, Math.max(0.7, cellW * 0.22), 0, Math.PI * 2);
        context.fill();
      }
    }
  }
}

function drawFogOfWar(context, explored, height, width, cellW, cellH) {
  if (!explored) return;
  context.fillStyle = 'rgba(5, 12, 18, 0.64)';
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      if (!explored[row]?.[col]) context.fillRect(col * cellW, row * cellH, cellW + 0.2, cellH + 0.2);
    }
  }
}

function drawHospitals(context, hospitals, cellW, cellH) {
  if (!hospitals) return;
  Object.values(hospitals).forEach((hospital) => {
    const x = hospital.col * cellW;
    const y = hospital.row * cellH;
    const cx = x + cellW / 2;
    const cy = y + cellH / 2;
    context.fillStyle = '#f7fbff';
    context.fillRect(x + cellW * 0.08, y + cellH * 0.08, cellW * 0.84, cellH * 0.84);
    context.fillStyle = '#ef5260';
    context.fillRect(cx - cellW * 0.3, cy - cellH * 0.08, cellW * 0.6, cellH * 0.16);
    context.fillRect(cx - cellW * 0.08, cy - cellH * 0.3, cellW * 0.16, cellH * 0.6);
  });
}

function drawVictims(context, victims, cellW, cellH, cellSize) {
  if (!victims) return;
  Object.values(victims).forEach((victim) => {
    if (victim.rescued) return;
    const x = victim.col * cellW + cellW / 2;
    const y = victim.row * cellH + cellH / 2;
    const color = SEVERITY_COLORS[victim.severity_name] || '#f9d6b3';
    if (victim.detected && !victim.assigned_agent) {
      const pulse = cellSize * (0.9 + Math.sin(Date.now() / 180) * 0.2);
      context.beginPath();
      context.arc(x, y, Math.max(3, pulse), 0, Math.PI * 2);
      context.strokeStyle = color;
      context.lineWidth = Math.max(0.7, cellSize * 0.11);
      context.stroke();
    }
    drawPerson(context, x, y, Math.max(2.8, cellSize * 0.72), color, victim.detected);
  });
}

function drawAgentPaths(context, agents, agentTrails, cellW, cellH) {
  if (!agents) return;
  Object.entries(agents).forEach(([agentId, agent]) => {
    const trail = agentTrails?.[agentId];
    if (trail?.length > 1) {
      context.beginPath();
      context.moveTo(trail[0].c * cellW + cellW / 2, trail[0].r * cellH + cellH / 2);
      trail.slice(1).forEach((point) => context.lineTo(point.c * cellW + cellW / 2, point.r * cellH + cellH / 2));
      context.strokeStyle = agent.type === 'drone' ? 'rgba(98, 232, 255, 0.38)' : agent.type === 'ambulance' ? 'rgba(255, 105, 118, 0.42)' : 'rgba(114, 236, 180, 0.34)';
      context.lineWidth = Math.max(1, cellW * 0.25);
      context.stroke();
    }

    if (agent.path?.length) {
      context.beginPath();
      context.moveTo(agent.col * cellW + cellW / 2, agent.row * cellH + cellH / 2);
      agent.path.forEach(([row, col]) => context.lineTo(col * cellW + cellW / 2, row * cellH + cellH / 2));
      context.strokeStyle = 'rgba(255, 255, 255, 0.34)';
      context.setLineDash([3, 3]);
      context.lineWidth = Math.max(0.7, cellW * 0.13);
      context.stroke();
      context.setLineDash([]);
    }
  });
}

function drawAgents(context, agents, cellW, cellH, cellSize) {
  if (!agents) return;
  Object.values(agents).forEach((agent) => {
    if (agent.type === 'traffic_light') return;
    const x = agent.col * cellW + cellW / 2;
    const y = agent.row * cellH + cellH / 2;
    const size = Math.max(3.1, cellSize * 0.76);
    if (agent.type === 'drone') drawDrone(context, x, y, size, agent.battery);
    else if (agent.type === 'ambulance') drawAmbulance(context, x, y, size, agent.carrying_victim);
    else drawGroundRobot(context, x, y, size);
  });
}

function drawPerson(context, x, y, size, shirtColor, detected) {
  const headRadius = size * 0.18;
  const bodyTop = y - size * 0.03;
  context.save();
  context.lineCap = 'round';
  context.strokeStyle = '#f2c4a0';
  context.lineWidth = Math.max(1, size * 0.13);
  context.beginPath();
  context.moveTo(x, bodyTop + size * 0.22);
  context.lineTo(x - size * 0.2, y + size * 0.35);
  context.moveTo(x, bodyTop + size * 0.22);
  context.lineTo(x + size * 0.2, y + size * 0.35);
  context.stroke();
  context.fillStyle = shirtColor;
  context.fillRect(x - size * 0.19, bodyTop, size * 0.38, size * 0.39);
  context.fillStyle = '#f2c4a0';
  context.beginPath();
  context.arc(x, y - size * 0.32, headRadius, 0, Math.PI * 2);
  context.fill();
  if (detected) {
    context.fillStyle = '#f9ffff';
    context.fillRect(x - size * 0.06, y - size * 0.36, size * 0.04, size * 0.04);
    context.fillRect(x + size * 0.03, y - size * 0.36, size * 0.04, size * 0.04);
  }
  context.restore();
}

function drawDrone(context, x, y, size, battery) {
  context.save();
  context.strokeStyle = '#a1f5ff';
  context.lineWidth = Math.max(1, size * 0.13);
  context.beginPath();
  context.moveTo(x - size * 0.46, y - size * 0.34);
  context.lineTo(x + size * 0.46, y + size * 0.34);
  context.moveTo(x + size * 0.46, y - size * 0.34);
  context.lineTo(x - size * 0.46, y + size * 0.34);
  context.stroke();
  context.fillStyle = '#0e3143';
  [[-0.48, -0.38], [0.48, -0.38], [-0.48, 0.38], [0.48, 0.38]].forEach(([px, py]) => {
    context.beginPath();
    context.arc(x + px * size, y + py * size, size * 0.18, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = '#62e8ff';
    context.lineWidth = Math.max(0.65, size * 0.06);
    context.stroke();
  });
  context.fillStyle = '#49cbe4';
  context.beginPath();
  context.moveTo(x, y - size * 0.34);
  context.lineTo(x + size * 0.3, y);
  context.lineTo(x, y + size * 0.34);
  context.lineTo(x - size * 0.3, y);
  context.closePath();
  context.fill();
  context.fillStyle = battery < 0.25 ? '#ff5f70' : '#cbfbff';
  context.fillRect(x - size * 0.07, y - size * 0.11, size * 0.14, size * 0.14);
  context.restore();
}

function drawAmbulance(context, x, y, size, carrying) {
  context.save();
  context.fillStyle = '#e9f4f8';
  context.fillRect(x - size * 0.5, y - size * 0.33, size, size * 0.68);
  context.fillStyle = '#e74b5d';
  context.fillRect(x - size * 0.5, y - size * 0.04, size, size * 0.18);
  context.fillRect(x - size * 0.12, y - size * 0.24, size * 0.24, size * 0.55);
  context.fillStyle = '#121a22';
  context.fillRect(x - size * 0.43, y + size * 0.28, size * 0.22, size * 0.12);
  context.fillRect(x + size * 0.21, y + size * 0.28, size * 0.22, size * 0.12);
  if (carrying) {
    context.strokeStyle = '#ffdd6a';
    context.lineWidth = Math.max(1, size * 0.1);
    context.strokeRect(x - size * 0.65, y - size * 0.48, size * 1.3, size * 0.96);
  }
  context.restore();
}

function drawGroundRobot(context, x, y, size) {
  context.save();
  context.fillStyle = '#3bd0a0';
  context.fillRect(x - size * 0.38, y - size * 0.3, size * 0.76, size * 0.65);
  context.fillStyle = '#102f38';
  context.fillRect(x - size * 0.2, y - size * 0.18, size * 0.4, size * 0.24);
  context.fillStyle = '#9df7ff';
  context.fillRect(x - size * 0.1, y - size * 0.12, size * 0.2, size * 0.06);
  context.restore();
}

function drawFlame(context, x, y, size) {
  context.save();
  context.fillStyle = '#ff612c';
  context.beginPath();
  context.moveTo(x, y - size * 1.1);
  context.quadraticCurveTo(x + size * 0.9, y - size * 0.12, x + size * 0.38, y + size * 0.5);
  context.quadraticCurveTo(x, y + size * 0.75, x - size * 0.4, y + size * 0.5);
  context.quadraticCurveTo(x - size * 0.85, y - size * 0.12, x, y - size * 1.1);
  context.fill();
  context.fillStyle = '#ffcf5c';
  context.beginPath();
  context.arc(x, y + size * 0.1, size * 0.25, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawSelectedCell(context, selectedCell, cellW, cellH) {
  if (!selectedCell) return;
  const x = selectedCell.c * cellW;
  const y = selectedCell.r * cellH;
  context.save();
  context.strokeStyle = '#a4f7ff';
  context.shadowBlur = 12;
  context.shadowColor = '#25d4ec';
  context.lineWidth = Math.max(1.5, Math.min(cellW, cellH) * 0.2);
  context.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
  context.restore();
}
