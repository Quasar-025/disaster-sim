import React, { useState } from 'react';
import { useDisasterState } from './hooks/useDisasterState';
import StrategicMap from './components/StrategicMap';
import TacticalView from './components/TacticalView';

function App() {
  const { worldState, connected, injectEarthquake, injectFlood } = useDisasterState();
  const [viewMode, setViewMode] = useState('2D');
  const [selectedCell, setSelectedCell] = useState(null);

  const handleCellClick = (r, c) => {
    setSelectedCell({ r, c });
  };

  const handleEarthquake = () => {
    if (selectedCell) {
      injectEarthquake(selectedCell.r, selectedCell.c, 7.5);
      setSelectedCell(null);
    }
  };

  const handleFlood = () => {
    if (selectedCell) {
      injectFlood(selectedCell.r, selectedCell.c, 100);
      setSelectedCell(null);
    }
  };

  return (
    <div style={{ backgroundColor: '#111', color: '#fff', minHeight: '100vh', padding: '20px', fontFamily: 'sans-serif' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1>Disaster Sim <span style={{ color: connected ? '#00ff00' : '#ff0000', fontSize: '0.5em' }}>{connected ? '● LIVE' : '○ OFFLINE'}</span></h1>
        
        <div>
          <button 
            onClick={() => setViewMode('2D')} 
            style={{ padding: '10px', background: viewMode === '2D' ? '#333' : '#111', color: '#fff', border: '1px solid #555' }}
          >
            Strategic 2D Map
          </button>
          <button 
            onClick={() => setViewMode('3D')} 
            style={{ padding: '10px', background: viewMode === '3D' ? '#333' : '#111', color: '#fff', border: '1px solid #555', marginLeft: '10px' }}
          >
            Tactical 3D View
          </button>
        </div>
      </header>

      <div style={{ display: 'flex', gap: '20px' }}>
        {/* Main Viewport */}
        <div style={{ flex: 1 }}>
          {viewMode === '2D' ? (
            <StrategicMap worldState={worldState} width={700} height={700} onCellClick={handleCellClick} />
          ) : (
            <TacticalView worldState={worldState} />
          )}
        </div>

        {/* Sidebar Controls */}
        <div style={{ width: '300px', backgroundColor: '#222', padding: '20px', borderRadius: '8px' }}>
          <h2>God Mode API</h2>
          <p style={{ fontSize: '0.9em', color: '#aaa' }}>
            Click a cell on the 2D map to select a target.
          </p>
          
          <div style={{ marginBottom: '20px', padding: '10px', background: '#333', borderRadius: '4px' }}>
            Target: {selectedCell ? `(${selectedCell.r}, ${selectedCell.c})` : 'None'}
          </div>

          <button 
            disabled={!selectedCell}
            onClick={handleEarthquake}
            style={{ display: 'block', width: '100%', padding: '10px', marginBottom: '10px', background: '#8a2b2b', color: 'white', border: 'none', cursor: selectedCell ? 'pointer' : 'not-allowed' }}
          >
            Trigger Earthquake
          </button>

          <button 
            disabled={!selectedCell}
            onClick={handleFlood}
            style={{ display: 'block', width: '100%', padding: '10px', background: '#1a5b7c', color: 'white', border: 'none', cursor: selectedCell ? 'pointer' : 'not-allowed' }}
          >
            Trigger Flash Flood
          </button>
          
          <div style={{ marginTop: '30px' }}>
            <h3>Telemetry</h3>
            {worldState ? (
              <ul style={{ listStyle: 'none', padding: 0, color: '#ccc', fontSize: '0.9em' }}>
                <li>Timestep: {worldState.timestep}</li>
                <li>Coverage: {(worldState.coverage * 100).toFixed(1)}%</li>
                <li>Victims Left: {worldState.victims_remaining}</li>
                <li>Rescued: {worldState.victims_rescued}</li>
                <li>Agents Active: {Object.keys(worldState.agents || {}).length}</li>
                <li>Avg Battery: {(worldState.avg_battery * 100).toFixed(1)}%</li>
              </ul>
            ) : (
              <p>Waiting for data...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
