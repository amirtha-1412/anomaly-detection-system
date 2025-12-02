import React, { useState, useEffect } from 'react';
import { Shield, Activity, AlertTriangle, CheckCircle, TrendingUp, Users, Clock, Server, Zap, Database, Terminal, ChevronRight } from 'lucide-react';

const API_URL = 'http://localhost:5000/api';

export default function AnomalyDetectionSystem() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [formData, setFormData] = useState({
    duration: 100,
    src_bytes: 250,
    dst_bytes: 180,
    wrong_fragment: 0,
    urgent: 0,
    hot: 0,
    num_failed_logins: 0,
    logged_in: 1,
    count: 15,
    srv_count: 12,
    serror_rate: 0.0,
    srv_serror_rate: 0.0,
    same_srv_rate: 1.0,
    diff_srv_rate: 0.0,
    protocol_type: 'tcp',
    service: 'http',
    flag: 'SF'
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);

  useEffect(() => {
    fetchSystemHealth();
    fetchStats();
  }, []);

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      setSystemHealth(data);
    } catch (error) {
      console.error('Error fetching health:', error);
      setSystemHealth({ status: 'offline', model_loaded: false });
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`);
      const data = await response.json();
      if (data.success) {
        setStats(data.stats);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
      setStats({
        dataset: { total_records: 0, anomalies: 0, normal: 0, anomaly_rate: 0 }
      });
    }
  };

  const predictAnomaly = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      
      if (data.success) {
        setPrediction(data);
        const newLog = {
          id: Date.now(),
          ...data,
          time: new Date().toLocaleTimeString()
        };
        setRecentLogs(prev => [newLog, ...prev.slice(0, 19)]);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error: Make sure the backend server is running on http://localhost:5000');
    }
    setLoading(false);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    const numFields = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
                       'num_failed_logins', 'logged_in', 'count', 'srv_count'];
    const floatFields = ['serror_rate', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate'];
    
    setFormData(prev => ({
      ...prev,
      [name]: numFields.includes(name) ? parseInt(value) || 0 : 
              floatFields.includes(name) ? parseFloat(value) || 0 : value
    }));
  };

  const getRiskColor = (level) => {
    switch(level) {
      case 'high': return 'from-red-500 to-red-600';
      case 'medium': return 'from-yellow-500 to-yellow-600';
      case 'low': return 'from-green-500 to-green-600';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  const getRiskBg = (level) => {
    switch(level) {
      case 'high': return 'bg-red-500/10 border-red-500/30';
      case 'medium': return 'bg-yellow-500/10 border-yellow-500/30';
      case 'low': return 'bg-green-500/10 border-green-500/30';
      default: return 'bg-gray-500/10 border-gray-500/30';
    }
  };

  const StatCard = ({ icon: Icon, title, value, subtitle, gradient }) => (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700 hover:border-slate-600 transition-all shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-lg bg-gradient-to-br ${gradient}`}>
          <Icon size={24} className="text-white" />
        </div>
      </div>
      <div>
        <p className="text-slate-400 text-sm font-medium mb-1">{title}</p>
        <p className="text-3xl font-bold text-white">{value}</p>
        {subtitle && <p className="text-slate-500 text-xs mt-2">{subtitle}</p>}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-pink-600/20 backdrop-blur-sm border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg">
                <Shield size={32} className="text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Anomaly Detection System</h1>
                <p className="text-slate-400 mt-1">KDD Cup 99 Dataset - Deep Neural Network</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className={`px-4 py-2 rounded-lg border ${systemHealth?.model_loaded ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${systemHealth?.model_loaded ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
                  <span className={`text-sm font-medium ${systemHealth?.model_loaded ? 'text-green-400' : 'text-red-400'}`}>
                    {systemHealth?.model_loaded ? 'Model Active' : 'Model Offline'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-900/50 border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Activity },
              { id: 'analyze', label: 'Analyze', icon: Terminal },
              { id: 'logs', label: 'Activity Logs', icon: Clock }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-all ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-400 bg-blue-500/10'
                    : 'border-transparent text-slate-400 hover:text-slate-300 hover:bg-slate-800/50'
                }`}
              >
                <tab.icon size={18} />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {stats?.dataset && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                  icon={Database}
                  title="Total Records"
                  value={stats.dataset.total_records.toLocaleString()}
                  subtitle="Network access logs"
                  gradient="from-blue-500 to-blue-600"
                />
                <StatCard
                  icon={AlertTriangle}
                  title="Anomalies"
                  value={stats.dataset.anomalies.toLocaleString()}
                  subtitle={`${stats.dataset.anomaly_rate}% detection rate`}
                  gradient="from-red-500 to-red-600"
                />
                <StatCard
                  icon={CheckCircle}
                  title="Normal Traffic"
                  value={stats.dataset.normal.toLocaleString()}
                  subtitle="Legitimate access"
                  gradient="from-green-500 to-green-600"
                />
                <StatCard
                  icon={Zap}
                  title="Model Accuracy"
                  value={stats.metadata?.final_val_accuracy ? `${(stats.metadata.final_val_accuracy * 100).toFixed(1)}%` : 'N/A'}
                  subtitle="Validation set"
                  gradient="from-purple-500 to-purple-600"
                />
              </div>
            )}

            <div className="grid lg:grid-cols-2 gap-6">
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Server size={20} className="text-blue-400" />
                  System Status
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-slate-950/50 rounded-lg">
                    <span className="text-slate-400">API Server</span>
                    <span className="text-green-400 font-semibold">Online</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-950/50 rounded-lg">
                    <span className="text-slate-400">Model Status</span>
                    <span className={`font-semibold ${systemHealth?.model_loaded ? 'text-green-400' : 'text-red-400'}`}>
                      {systemHealth?.model_loaded ? 'Loaded' : 'Not Loaded'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-950/50 rounded-lg">
                    <span className="text-slate-400">Features</span>
                    <span className="text-blue-400 font-semibold">{stats?.metadata?.num_features || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-950/50 rounded-lg">
                    <span className="text-slate-400">Training Samples</span>
                    <span className="text-purple-400 font-semibold">{stats?.metadata?.train_samples?.toLocaleString() || 'N/A'}</span>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-blue-600/10 via-purple-600/10 to-pink-600/10 rounded-xl p-6 border border-blue-500/30">
                <h3 className="text-xl font-bold text-white mb-4">Quick Start</h3>
                <p className="text-slate-300 mb-6">
                  Use the Analyze tab to test network traffic patterns and detect potential security threats using our trained deep neural network model.
                </p>
                <button
                  onClick={() => setActiveTab('analyze')}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all flex items-center justify-center gap-2 shadow-lg"
                >
                  Start Analysis
                  <ChevronRight size={20} />
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analyze' && (
          <div className="grid lg:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Terminal size={24} className="text-blue-400" />
                Network Traffic Parameters
              </h2>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Duration (ms)
                    </label>
                    <input
                      type="number"
                      name="duration"
                      value={formData.duration}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Failed Logins
                    </label>
                    <input
                      type="number"
                      name="num_failed_logins"
                      min="0"
                      value={formData.num_failed_logins}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Source Bytes
                    </label>
                    <input
                      type="number"
                      name="src_bytes"
                      value={formData.src_bytes}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Destination Bytes
                    </label>
                    <input
                      type="number"
                      name="dst_bytes"
                      value={formData.dst_bytes}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Protocol
                    </label>
                    <select
                      name="protocol_type"
                      value={formData.protocol_type}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="tcp">TCP</option>
                      <option value="udp">UDP</option>
                      <option value="icmp">ICMP</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Service
                    </label>
                    <select
                      name="service"
                      value={formData.service}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="http">HTTP</option>
                      <option value="ftp">FTP</option>
                      <option value="smtp">SMTP</option>
                      <option value="ssh">SSH</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Flag
                    </label>
                    <select
                      name="flag"
                      value={formData.flag}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="SF">SF</option>
                      <option value="S0">S0</option>
                      <option value="REJ">REJ</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Connection Count
                    </label>
                    <input
                      type="number"
                      name="count"
                      value={formData.count}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Service Count
                    </label>
                    <input
                      type="number"
                      name="srv_count"
                      value={formData.srv_count}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Error Rate (0-1)
                    </label>
                    <input
                      type="number"
                      name="serror_rate"
                      min="0"
                      max="1"
                      step="0.1"
                      value={formData.serror_rate}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Same Service Rate
                    </label>
                    <input
                      type="number"
                      name="same_srv_rate"
                      min="0"
                      max="1"
                      step="0.1"
                      value={formData.same_srv_rate}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <button
                  onClick={predictAnomaly}
                  disabled={loading || !systemHealth?.model_loaded}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      Analyzing Traffic...
                    </>
                  ) : (
                    <>
                      <Zap size={20} />
                      Run Detection
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <TrendingUp size={24} className="text-purple-400" />
                Detection Results
              </h2>

              {!prediction ? (
                <div className="flex flex-col items-center justify-center h-96 text-slate-500">
                  <Shield size={64} strokeWidth={1} />
                  <p className="mt-4 text-lg">Run analysis to see results</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className={`p-6 rounded-xl border-2 ${prediction.is_anomaly ? 'bg-red-500/10 border-red-500/50' : 'bg-green-500/10 border-green-500/50'}`}>
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-lg font-semibold text-white">Detection Status</span>
                      {prediction.is_anomaly ? (
                        <AlertTriangle className="text-red-400" size={32} />
                      ) : (
                        <CheckCircle className="text-green-400" size={32} />
                      )}
                    </div>
                    <p className={`text-3xl font-bold ${prediction.is_anomaly ? 'text-red-400' : 'text-green-400'}`}>
                      {prediction.is_anomaly ? 'THREAT DETECTED' : 'NORMAL TRAFFIC'}
                    </p>
                  </div>

                  <div className="bg-slate-950/50 p-4 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium text-slate-300">Confidence Score</span>
                      <span className="text-sm font-bold text-white">{prediction.confidence}%</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
                      <div
                        className={`h-3 bg-gradient-to-r ${getRiskColor(prediction.risk_level)} transition-all duration-500`}
                        style={{ width: `${prediction.confidence}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="bg-slate-950/50 p-4 rounded-lg">
                    <p className="text-sm font-medium text-slate-300 mb-3">Risk Assessment</p>
                    <div className={`px-4 py-3 rounded-lg bg-gradient-to-r ${getRiskColor(prediction.risk_level)} text-white font-bold text-center text-lg`}>
                      {prediction.risk_level.toUpperCase()} RISK
                    </div>
                  </div>

                  {prediction.threat_indicators && prediction.threat_indicators.length > 0 && (
                    <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                      <p className="font-semibold text-yellow-400 mb-3 flex items-center gap-2">
                        <AlertTriangle size={18} />
                        Threat Indicators
                      </p>
                      <ul className="space-y-2">
                        {prediction.threat_indicators.map((indicator, idx) => (
                          <li key={idx} className="text-sm text-yellow-300 flex items-start gap-2">
                            <span className="text-yellow-500 mt-1">•</span>
                            <span>{indicator}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {prediction.is_anomaly && (
                    <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                      <p className="font-semibold text-red-400 mb-2">Recommended Actions</p>
                      <ul className="text-sm text-red-300 space-y-1 ml-4 list-disc">
                        <li>Immediately block suspicious IP address</li>
                        <li>Review access logs for similar patterns</li>
                        <li>Escalate to security operations center</li>
                        <li>Monitor for additional malicious activity</li>
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-centergap-2">
              <Clock size={24} className="text-blue-400" />
              Recent Activity Log
            </h2>

            {recentLogs.length === 0 ? (
              <div className="text-center py-16 text-slate-500">
                <Activity size={48} strokeWidth={1} className="mx-auto mb-4 opacity-50" />
                <p className="text-lg">No activity logs yet</p>
                <p className="text-sm mt-2">Run some analyses to populate the log</p>
              </div>
            ) : (
              <div className="space-y-3">
                {recentLogs.map(log => (
                  <div key={log.id} className="bg-slate-950/50 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition-all">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm text-slate-400">{log.time}</span>
                      <span className={`px-3 py-1 rounded-full text-xs font-bold text-white bg-gradient-to-r ${getRiskColor(log.risk_level)}`}>
                        {log.risk_level.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 mb-2">
                      {log.is_anomaly ? (
                        <span className="text-red-400 text-sm font-semibold flex items-center gap-2">
                          <AlertTriangle size={16} />
                          Anomaly ({log.confidence}%)
                        </span>
                      ) : (
                        <span className="text-green-400 text-sm font-semibold flex items-center gap-2">
                          <CheckCircle size={16} />
                          Normal ({log.confidence}%)
                        </span>
                      )}
                    </div>
                    {log.threat_indicators && log.threat_indicators.length > 0 && (
                      <div className="mt-2 text-xs text-slate-400 bg-slate-900/50 p-2 rounded">
                        <span className="font-semibold text-yellow-400">Indicators: </span>
                        {log.threat_indicators.join(', ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}