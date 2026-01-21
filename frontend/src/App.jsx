import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, ResponsiveContainer } from 'recharts';
import './App.css';

const FluorideDashboard = () => {
  const [pH, setPH] = useState('7.9');
  const [ec, setEC] = useState('395');
  const [hardness, setHardness] = useState('160');
  const [prediction, setPrediction] = useState(0.53);
  const [isLoading, setIsLoading] = useState(false);

  // Training loss data - Replace with your actual training history
  const lossData = [
    { epoch: 0, train: 0.6, validation: 0.58 },
    { epoch: 5, train: 0.35, validation: 0.33 },
    { epoch: 10, train: 0.28, validation: 0.27 },
    { epoch: 15, train: 0.25, validation: 0.24 },
    { epoch: 20, train: 0.23, validation: 0.23 },
    { epoch: 25, train: 0.22, validation: 0.22 },
    { epoch: 30, train: 0.21, validation: 0.21 },
    { epoch: 35, train: 0.205, validation: 0.205 },
    { epoch: 40, train: 0.2, validation: 0.2 },
    { epoch: 45, train: 0.195, validation: 0.195 },
    { epoch: 50, train: 0.19, validation: 0.19 }
  ];

  // Feature correlation data - Replace with your actual correlations
  const correlationData = [
    { feature: 'pH', correlation: 0.32 },
    { feature: 'EC', correlation: 0.47 },
    { feature: 'Hardness', correlation: 0.21 }
  ];

  const getRiskLevel = (value) => {
    if (value < 1.0) return { text: 'Safe', color: '#10b981' };
    if (value <= 1.5) return { text: 'Moderate', color: '#fbbf24' };
    return { text: 'Unsafe', color: '#ef4444' };
  };

  const handlePredict = async () => {
    setIsLoading(true);
    
    try {
      // Call the Flask API endpoint
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          pH: parseFloat(pH), 
          ec: parseFloat(ec), 
          hardness: parseFloat(hardness) 
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }
      
      const data = await response.json();
      setPrediction(data.fluoride);
      
      console.log('Prediction successful:', data);
      
    } catch (error) {
      console.error('Error predicting:', error);
      alert(`Prediction failed: ${error.message}\n\nMake sure the Flask backend is running on http://localhost:5000`);
    } finally {
      setIsLoading(false);
    }
  };

  const riskLevel = getRiskLevel(prediction);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-blue-50 to-cyan-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <svg className="w-12 h-12 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>
            </svg>
            <h1 className="text-4xl font-bold text-gray-800">Fluoride Prediction Dashboard</h1>
          </div>
        </div>

        {/* Top Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Input Panel */}
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-6">Enter Water Parameters</h2>
            
            <div className="space-y-6">
              <div>
                <label className="block text-gray-700 font-medium mb-2">pH</label>
                <input
                  type="number"
                  step="0.1"
                  value={pH}
                  onChange={(e) => setPH(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition"
                />
              </div>

              <div>
                <label className="block text-gray-700 font-medium mb-2">EC (µS/cm)</label>
                <input
                  type="number"
                  value={ec}
                  onChange={(e) => setEC(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition"
                />
              </div>

              <div>
                <label className="block text-gray-700 font-medium mb-2">Hardness (mg/L)</label>
                <input
                  type="number"
                  value={hardness}
                  onChange={(e) => setHardness(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition"
                />
              </div>

              <button
                onClick={handlePredict}
                disabled={isLoading}
                className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Predicting...' : 'Predict'}
              </button>
            </div>
          </div>

          {/* Prediction Result */}
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-6">
              Predicted Fluoride: <span className="text-blue-600">{prediction.toFixed(2)} mg/L</span>
            </h2>

            {/* Horizontal Bar Gauge */}
            <div className="w-full max-w-md mx-auto mb-6">
              {/* Bar container */}
              <div className="relative h-16 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                {/* Color segments */}
                <div className="absolute inset-0 flex">
                  {/* Safe - Green (0 to 1.0) - 50% */}
                  <div className="h-full bg-gradient-to-r from-green-400 to-green-500" style={{ width: '50%' }}></div>
                  {/* Moderate - Yellow (1.0 to 1.5) - 30% */}
                  <div className="h-full bg-gradient-to-r from-yellow-400 to-yellow-500" style={{ width: '30%' }}></div>
                  {/* Unsafe - Red (above 1.5) - 20% */}
                  <div className="h-full bg-gradient-to-r from-red-400 to-red-500" style={{ width: '20%' }}></div>
                </div>
                
                {/* Needle indicator */}
                <div 
                  className="absolute top-0 bottom-0 w-1 bg-blue-900 shadow-lg transition-all duration-500"
                  style={{ left: `${Math.min((prediction / 2.0) * 100, 100)}%`, transform: 'translateX(-50%)' }}
                >
                  {/* Arrow pointer */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                    <div className="w-6 h-6 bg-blue-900 rotate-45 shadow-lg"></div>
                  </div>
                </div>
              </div>
              
              {/* Scale markers */}
              <div className="relative mt-2 px-2">
                <div className="flex justify-between text-sm font-medium text-gray-600">
                  <span>0</span>
                  <span>0.5</span>
                  <span>1.0</span>
                  <span>1.5</span>
                  <span>2.0+</span>
                </div>
                {/* Tick marks */}
                <div className="absolute -top-3 left-0 right-0 flex justify-between px-2">
                  <div className="w-0.5 h-2 bg-gray-400"></div>
                  <div className="w-0.5 h-2 bg-gray-400"></div>
                  <div className="w-0.5 h-2 bg-gray-400"></div>
                  <div className="w-0.5 h-2 bg-gray-400"></div>
                  <div className="w-0.5 h-2 bg-gray-400"></div>
                </div>
              </div>
            </div>

            {/* Value Display */}
            <div className="text-center mb-6">
              <div className="text-5xl font-bold text-gray-800 mb-2">{prediction.toFixed(2)}</div>
              <div className="text-lg">
                Risk Level: <span className="font-semibold" style={{ color: riskLevel.color }}>{riskLevel.text}</span>
              </div>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-4 text-sm flex-wrap">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-green-500"></div>
                <span>Safe (below 1.0)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-yellow-400"></div>
                <span>Moderate (1.0 - 1.5 mg/L)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-red-500"></div>
                <span>Unsafe (above 1.5 mg/L)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Training Loss Curve */}
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-6">Training Loss Curve</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} domain={[0, 0.6]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="train" stroke="#3b82f6" strokeWidth={3} name="Train Loss" dot={{ r: 5, fill: '#3b82f6' }} />
                <Line type="monotone" dataKey="validation" stroke="#10b981" strokeWidth={3} name="Validation Loss" dot={{ r: 5, fill: '#10b981' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Feature Correlation */}
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-6">Feature Correlation with Fluoride</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={correlationData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="feature" label={{ value: 'Feature', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Correlation', angle: -90, position: 'insideLeft' }} domain={[0, 0.5]} />
                <Tooltip />
                <Bar dataKey="correlation" fill="#60a5fa" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FluorideDashboard;