import React, { useState } from 'react';
import { useDisasterState } from './hooks/useDisasterState';
import StrategicMap from './components/StrategicMap';
import TacticalView from './components/TacticalView';
import StatsPanel from './components/StatsPanel';
import './App.css';

function App() {
  const { 
    worldState, 
    connected, 
    events, 
    agentTrails, 
    setSimulationSpeed, 
    injectEarthquake, 
    injectFlood 
  } = useDisasterState();
  
  const [viewMode, setViewMode] = useState('2D');
  const [selectedCell, setSelectedCell] = useState(null);
  const [speed, setSpeed] = useState(20);

  const handleCellClick = (r, c) => {
    setSelectedCell({ r, c });
  };

  const handleSpeedChange = (e) => {
    const newSpeed = Number(e.target.value);
    setSpeed(newSpeed);
    setSimulationSpeed(newSpeed);
  };

  return (
    <div className="app-container">
      
      {/* Top Navigation / Header */}
      <header className="header glass-panel">
        <h1 className="header-title">
          Swarm Intelligence Command Center
        </h1>
        
        <div className="controls-group">
          <div className="slider-container">
            <label htmlFor="speed-slider">Sim Speed: {speed}Hz</label>
            <input 
              id="speed-slider"
              type="range" 
              min="1" max="50" 
              value={speed} 
              onChange={handleSpeedChange} 
              className="speed-slider"
            />
          </div>

          <div style={{ display: 'flex', gap: '8px', marginLeft: '16px' }}>
            <button 
              onClick={() => setViewMode('2D')} 
              style={{ 
                padding: '6px 12px', 
                background: viewMode === '2D' ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255, 255, 255, 0.05)', 
                color: viewMode === '2D' ? '#06b6d4' : '#94a3b8', 
                border: `1px solid ${viewMode === '2D' ? 'rgba(6, 182, 212, 0.5)' : 'transparent'}`,
                borderRadius: '6px',
                cursor: 'pointer'
              }}
            >
              2D Strategic
            </button>
            <button 
              onClick={() => setViewMode('3D')} 
              style={{ 
                padding: '6px 12px', 
                background: viewMode === '3D' ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255, 255, 255, 0.05)', 
                color: viewMode === '3D' ? '#06b6d4' : '#94a3b8', 
                border: `1px solid ${viewMode === '3D' ? 'rgba(6, 182, 212, 0.5)' : 'transparent'}`,
                borderRadius: '6px',
                cursor: 'pointer'
              }}
            >
              3D Tactical
            </button>
          </div>
          
          <div className="status-indicator">
            <div className={`status-dot ${connected ? 'status-live' : 'status-offline'}`}></div>
            <span>{connected ? 'LIVE' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <div className="main-content">
        
        {/* Center: Map Viewport */}
        <div className="map-container">
          <div className="map-wrapper">
            {viewMode === '2D' ? (
              <StrategicMap 
                worldState={worldState} 
                agentTrails={agentTrails}
                onCellClick={handleCellClick} 
              />
            ) : (
              <TacticalView worldState={worldState} />
            )}
          </div>
          
          {/* Quick Disaster Controls Overlay */}
          <div style={{ position: 'absolute', bottom: '16px', left: '16px', right: '16px', display: 'flex', gap: '16px', pointerEvents: 'none' }}>
            <div className="glass-panel" style={{ padding: '12px', pointerEvents: 'auto', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span className="text-muted" style={{ fontSize: '0.85rem' }}>God Mode Injection:</span>
              <span style={{ fontSize: '0.85rem', color: selectedCell ? '#fff' : '#666', fontFamily: 'var(--font-mono)' }}>
                Target: {selectedCell ? `[${selectedCell.r}, ${selectedCell.c}]` : '[ Select cell on map ]'}
              </span>
              
              <button 
                disabled={!selectedCell}
                onClick={() => { injectEarthquake(selectedCell.r, selectedCell.c, 7.5); setSelectedCell(null); }}
                style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.5)', padding: '6px 16px', borderRadius: '4px', cursor: selectedCell ? 'pointer' : 'not-allowed', opacity: selectedCell ? 1 : 0.5 }}
              >
                Earthquake
              </button>

              <button 
                disabled={!selectedCell}
                onClick={() => { injectFlood(selectedCell.r, selectedCell.c, 100); setSelectedCell(null); }}
                style={{ background: 'rgba(6, 182, 212, 0.2)', color: '#06b6d4', border: '1px solid rgba(6, 182, 212, 0.5)', padding: '6px 16px', borderRadius: '4px', cursor: selectedCell ? 'pointer' : 'not-allowed', opacity: selectedCell ? 1 : 0.5 }}
              >
                Flash Flood
              </button>
            </div>
            
            <div style={{ flex: 1 }}></div>
            
            <div className="glass-panel" style={{ padding: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
               <span className="text-muted mono" style={{ fontSize: '0.9rem' }}>
                  {worldState ? `T+${worldState.timestep}` : 'T+0'}
               </span>
            </div>
          </div>
        </div>

        {/* Right: Stats & Feed */}
        <div className="side-panel">
          <StatsPanel worldState={worldState} events={events} />
        </div>
        
      </div>
    </div>
  );
}

export default App;
