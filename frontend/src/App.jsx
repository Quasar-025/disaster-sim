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
    injectFlood,
  } = useDisasterState();

  const [viewMode, setViewMode] = useState('2D');
  const [selectedCell, setSelectedCell] = useState(null);
  const [speed, setSpeed] = useState(20);

  const handleSpeedChange = (event) => {
    const newSpeed = Number(event.target.value);
    setSpeed(newSpeed);
    setSimulationSpeed(newSpeed);
  };

  const injectAtSelection = (action) => {
    if (!selectedCell) return;
    action(selectedCell.r, selectedCell.c);
    setSelectedCell(null);
  };

  return (
    <div className="app-container">
      <header className="header glass-panel">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div>
            <p className="eyebrow">Autonomous emergency response</p>
            <h1 className="header-title">Swarm Command</h1>
          </div>
        </div>

        <div className="controls-group">
          <div className="slider-container">
            <label htmlFor="speed-slider">Simulation <strong>{speed} Hz</strong></label>
            <input
              id="speed-slider"
              type="range"
              min="1"
              max="50"
              value={speed}
              onChange={handleSpeedChange}
              className="speed-slider"
            />
          </div>

          <div className="view-mode-switcher" aria-label="Map view">
            <button
              type="button"
              className={viewMode === '2D' ? 'is-active' : ''}
              onClick={() => setViewMode('2D')}
            >
              <span>▦</span> Strategic map
            </button>
            <button
              type="button"
              className={viewMode === '3D' ? 'is-active' : ''}
              onClick={() => setViewMode('3D')}
            >
              <span>◇</span> Drone cameras
            </button>
          </div>

          <div className="status-indicator" aria-label={connected ? 'Live connection' : 'Offline connection'}>
            <div className={`status-dot ${connected ? 'status-live' : 'status-offline'}`} />
            <span>{connected ? 'LIVE' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="map-container" aria-label={viewMode === '2D' ? 'Strategic response map' : 'Three dimensional tactical map'}>
          <div className="map-header">
            <div>
              <span className="panel-kicker">{viewMode === '2D' ? 'Situation awareness' : 'Aerial reconnaissance'}</span>
              <h2>{viewMode === '2D' ? 'Response grid' : 'Live city digital twin'}</h2>
            </div>
            <div className="map-meta">
              <span className="map-meta-dot" />
              {viewMode === '2D' ? 'Click a cell to target an event' : 'Select a drone for forward POV'}
            </div>
          </div>

          <div className="map-wrapper">
            {viewMode === '2D' ? (
              <StrategicMap
                worldState={worldState}
                agentTrails={agentTrails}
                selectedCell={selectedCell}
                onCellClick={(r, c) => setSelectedCell({ r, c })}
              />
            ) : (
              <TacticalView worldState={worldState} agentTrails={agentTrails} />
            )}
          </div>

          <div className="operation-bar">
            <div className="target-readout">
              <span className="target-icon">⌖</span>
              <div>
                <span className="panel-kicker">Event target</span>
                <strong>{selectedCell ? `GRID ${selectedCell.r.toString().padStart(3, '0')} · ${selectedCell.c.toString().padStart(3, '0')}` : 'Select a map cell'}</strong>
              </div>
            </div>

            <div className="disaster-actions">
              <button
                type="button"
                className="action-button action-earthquake"
                disabled={!selectedCell}
                onClick={() => injectAtSelection((r, c) => injectEarthquake(r, c, 7.5))}
              >
                <span>⌁</span> Earthquake
              </button>
              <button
                type="button"
                className="action-button action-flood"
                disabled={!selectedCell}
                onClick={() => injectAtSelection((r, c) => injectFlood(r, c, 100))}
              >
                <span>≈</span> Flash flood
              </button>
            </div>

            <div className="world-clock mono">T+{worldState?.timestep ?? 0}</div>
          </div>
        </section>

        <aside className="side-panel" aria-label="Response telemetry">
          <StatsPanel worldState={worldState} events={events} />
        </aside>
      </main>
    </div>
  );
}

export default App;
