import React from 'react';

export default function StatsPanel({ worldState, events }) {
  if (!worldState) {
    return <div className="glass-panel" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Connecting to Command Center...</div>;
  }

  const { victims_remaining, victims_rescued, victims_detected, victims_transported, coverage, agents, victims } = worldState;

  // Fleet stats
  const fleetCounts = { drone: 0, ambulance: 0, ground_robot: 0 };
  let ambulancesRescuing = 0;
  
  Object.values(agents).forEach(a => {
    if (fleetCounts[a.type] !== undefined) fleetCounts[a.type]++;
    if (a.type === 'ambulance' && a.carrying_victim) ambulancesRescuing++;
  });

  // Triage stats
  let severityCounts = { critical: 0, serious: 0, stable: 0 };
  Object.values(victims).forEach(v => {
    if (!v.rescued) severityCounts[v.severity_name]++;
  });

  const coveragePercent = (coverage * 100).toFixed(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '16px' }}>
      
      {/* Triage Overview */}
      <div className="glass-panel">
        <h3 className="text-muted" style={{ marginBottom: '16px', fontSize: '0.85rem', textTransform: 'uppercase' }}>Victim Triage Status</h3>
        
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-value text-cyan">{victims_detected}</span>
            <span className="stat-label">Detected</span>
          </div>
          <div className="stat-card">
            <span className="stat-value text-purple">{victims_rescued}</span>
            <span className="stat-label">Rescued</span>
          </div>
          <div className="stat-card">
            <span className="stat-value text-green">{victims_transported}</span>
            <span className="stat-label">Transported</span>
          </div>
          <div className="stat-card">
            <span className="stat-value text-orange">{victims_remaining}</span>
            <span className="stat-label">Remaining</span>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginTop: '12px' }}>
          <span style={{ color: '#ef4444' }}>● Critical: {severityCounts.critical}</span>
          <span style={{ color: '#f97316' }}>● Serious: {severityCounts.serious}</span>
          <span style={{ color: '#eab308' }}>● Stable: {severityCounts.stable}</span>
        </div>
      </div>

      {/* Fleet Overview */}
      <div className="glass-panel">
        <h3 className="text-muted" style={{ marginBottom: '16px', fontSize: '0.85rem', textTransform: 'uppercase' }}>Swarm Fleet Operations</h3>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', fontSize: '0.85rem' }}>
          <span>Drones (<span className="text-cyan">{fleetCounts.drone}</span> active)</span>
          <span className="text-muted">Coverage: {coveragePercent}%</span>
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', fontSize: '0.85rem' }}>
          <span>Ambulances (<span className="text-red">{fleetCounts.ambulance}</span> active)</span>
          <span className="text-muted">{ambulancesRescuing} en route to hospital</span>
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
          <span>Fleet Power</span>
          <span className="text-green mono">{(worldState.avg_battery * 100).toFixed(0)}%</span>
        </div>
        <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', marginTop: '4px' }}>
          <div style={{ width: `${worldState.avg_battery * 100}%`, height: '100%', background: 'var(--accent-green)', borderRadius: '2px' }}></div>
        </div>
      </div>

      {/* Event Feed */}
      <div className="event-feed glass-panel" style={{ flex: 1, padding: 0 }}>
        <div style={{ padding: '12px', borderBottom: '1px solid rgba(255,255,255,0.05)', backgroundColor: 'rgba(0,0,0,0.2)' }}>
          <h3 className="text-muted" style={{ margin: 0, fontSize: '0.85rem', textTransform: 'uppercase' }}>Intelligence Feed</h3>
        </div>
        
        <div style={{ padding: '12px', overflowY: 'auto', flex: 1 }}>
          {events.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontStyle: 'italic', textAlign: 'center', marginTop: '20px' }}>
              Waiting for swarm telemetry...
            </div>
          ) : (
            <ul className="event-feed-list">
              {events.map((ev, i) => (
                <li key={ev.id || i} className={`event-item type-${ev.type}`}>
                  <span className="event-time">{ev.time}</span>
                  {ev.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
