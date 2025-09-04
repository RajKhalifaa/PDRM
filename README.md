# 🏛️ PDRM Asset Management Dashboard

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/xxxraj/pdrm-asset-dashboard)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)](https://python.org)
[![AI](https://img.shields.io/badge/AI-Powered-purple?logo=openai)](https://openai.com)

A comprehensive asset management solution for the **Royal Malaysia Police (PDRM)** featuring advanced analytics, predictive maintenance, AI-powered insights, and professional reporting capabilities.

## 🚀 **Latest Updates (v1.2)**

### ✨ **New Features Added:**

- 🤖 **AI Assistant Chatbot** with conversational asset analysis
- 📊 **Enhanced Dashboard UI** with professional blue gradient theme
- 🐳 **Docker Containerization** ready for ECS deployment
- 📋 **Malaysian Government Reports** (KEW.PS-6 format)
- 🔮 **Advanced Predictive Analytics** with 6-12 month forecasting
- 🎨 **Professional Styling** with consistent white text on blue backgrounds
- 🖥️ **ECS-Optimized Deployment** with improved Kaleido handling

### 🔧 **Bug Fixes:**

- ✅ Fixed Kaleido/Chrome dependency issues for container deployment
- ✅ Resolved chart display problems in ECS environments
- ✅ Improved chat message text color visibility
- ✅ Enhanced startup process with cleaner initialization

## 📊 **Dashboard Overview**

The PDRM Asset Management Dashboard provides three main analytical modules:

### 1. 📈 **Descriptive Analytics**

- **Real-time KPIs** with interactive charts
- **Asset distribution** by category, state, and condition
- **Maintenance priority analysis** with color-coded urgency levels
- **Geographic heatmaps** showing asset distribution across Malaysia
- **Category breakdown** with detailed asset type analysis

### 2. 🔮 **Predictive Analytics**

- **Machine Learning-powered** condition prediction (89.2% accuracy)
- **Maintenance demand forecasting** for 3-12 months ahead
- **Risk assessment** by asset category and type
- **Resource planning insights** with workload distribution
- **Proactive maintenance scheduling** recommendations

### 3. 🤖 **AI Report Generator**

- **OpenAI-powered** comprehensive report generation
- **Professional Word documents** with embedded charts
- **Executive summary** and strategic recommendations
- **Malaysian government format** reports (KEW.PS-6)
- **Multi-format export** (Word, PDF, Text)

## 🛠️ **Technology Stack**

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python 3.11 with FastAPI integration
- **Machine Learning**: scikit-learn, XGBoost with enhanced preprocessing
- **Visualization**: Plotly for interactive charts
- **AI Integration**: OpenAI GPT-4 for intelligent insights
- **Containerization**: Docker with multi-stage builds
- **Deployment**: AWS ECS-ready with health checks
- **Document Generation**: python-docx, FPDF2

## 🐳 **Docker Deployment**

### **Quick Start with Docker Hub**

```bash
# Pull and run the latest version
docker pull xxxraj/pdrm-asset-dashboard:latest
docker run -p 8501:8501 xxxraj/pdrm-asset-dashboard:latest

# Access the dashboard
# Local: http://localhost:8501
# Network: http://your-server-ip:8501
```

### **Available Docker Images**

| Version  | Description                      | Size  | Status              |
| -------- | -------------------------------- | ----- | ------------------- |
| `latest` | Latest stable version (v1.2)     | 944MB | ✅ Production Ready |
| `v1.2`   | ECS-optimized with Kaleido fixes | 944MB | ✅ Recommended      |
| `v1.1`   | Enhanced features version        | 944MB | ✅ Stable           |
| `v1.0`   | Initial release version          | 944MB | ✅ Legacy           |

### **AWS ECS Deployment**

```bash
# Task Definition for ECS
{
  "family": "pdrm-dashboard",
  "taskRoleArn": "arn:aws:iam::account:role/taskRole",
  "executionRoleArn": "arn:aws:iam::account:role/executionRole",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "pdrm-dashboard",
      "image": "xxxraj/pdrm-asset-dashboard:v1.2",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-openai-api-key"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "python3 /app/healthcheck.py"
        ],
        "interval": 30,
        "timeout": 30,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

## 🗂️ **Project Structure**

```
PDRM 3/
├── 📊 Dashboard & UI
│   ├── streamlit_dashboard.py          # Main dashboard application
│   ├── assets/                         # Static assets (logos, images)
│   └── styles/                         # Custom CSS styling
│
├── 🤖 Machine Learning
│   ├── train_ml_models_enhanced.py     # Enhanced ML training pipeline
│   ├── enhanced_model/                 # Trained models directory
│   │   ├── enhanced_pdrm_asset_condition_model.joblib
│   │   ├── enhanced_label_encoder.joblib
│   │   └── enhanced_model_info.json
│   └── demo_enhanced_predictions.py    # ML prediction demos
│
├── 📋 Data & Reports
│   ├── assets_data.csv                 # Asset dataset (20,000 records)
│   ├── generate_data.py                # Data generation script
│   └── reports/                        # Generated reports directory
│
├── 🐳 Deployment
│   ├── Dockerfile                      # Multi-stage Docker build
│   ├── docker-compose.yml              # Local development setup
│   ├── entrypoint.sh                   # Container initialization script
│   ├── healthcheck.py                  # Health check endpoint
│   └── requirements.txt                # Python dependencies
│
├── 📚 Documentation
│   ├── README.md                       # This file
│   ├── DASHBOARD_GUIDE.md              # User guide for dashboard
│   ├── ENHANCEMENT_SUMMARY.md          # Feature enhancement details
│   ├── DEPLOYMENT.md                   # Deployment instructions
│   └── DOCKER_DEPLOYMENT.md            # Docker-specific deployment guide
│
└── 🔧 Scripts & Tools
    ├── build-and-push.ps1              # Windows Docker build script
    ├── build-and-push.sh               # Linux Docker build script
    └── test_word_export.py             # Report generation testing
```

## 📊 **Dataset Overview**

### **Scale & Coverage**

- **20,000 asset records** across Malaysia
- **4 major categories**: Weapons, ICT Assets, Vehicles, Devices
- **14 Malaysian states** represented
- **Realistic Malaysian context**: Malay names, license plates, locations

### **Asset Categories Distribution**

| Category          | Count | Example Types                           |
| ----------------- | ----- | --------------------------------------- |
| 🔫 **Weapons**    | 5,000 | Glock 19, M4 Carbine, HK416, Tasers     |
| 💻 **ICT Assets** | 5,000 | Dell Laptops, HP Servers, Cisco Routers |
| 🚗 **Vehicles**   | 5,000 | Honda Civic Patrol, Yamaha FZ150i       |
| 📱 **Devices**    | 5,000 | Body Cameras, CCTV Systems, Drones      |

### **Predictive Features (28 total)**

- **Basic Info**: Category, Type, State, City, Purchase Date
- **Usage Metrics**: Hours, Mileage, Maintenance History
- **Condition Indicators**: Current Status, Risk Factors
- **Operational Data**: Assigned Officer, Warranty Status

## 🧠 **Machine Learning Performance**

### **Enhanced Model (v2.0)**

- **Algorithm**: Enhanced Random Forest with feature engineering
- **Accuracy**: 89.2% (improved from 34.9%)
- **F1-Score**: 0.891 (weighted average)
- **Features**: 28 engineered features with domain expertise

### **Performance by Category**

| Asset Category | Precision | Recall | F1-Score |
| -------------- | --------- | ------ | -------- |
| Weapons        | 0.91      | 0.89   | 0.90     |
| ICT Assets     | 0.88      | 0.91   | 0.89     |
| Vehicles       | 0.92      | 0.87   | 0.89     |
| Devices        | 0.87      | 0.90   | 0.88     |

### **Condition Prediction Classes**

- ✅ **Good**: Assets in optimal condition (33.4%)
- ⚠️ **Needs Action**: Requires maintenance attention (33.3%)
- ❌ **Damaged**: Critical condition requiring immediate action (33.3%)

## 🤖 **AI Assistant Features**

### **Conversational Analytics**

- **Natural language queries** about asset data
- **Pre-defined prompts** for common analyses
- **Real-time insights** based on current data
- **Contextual recommendations** for asset management

### **Sample Queries**

- "Show me the overall asset status for PDRM"
- "Which assets need immediate maintenance?"
- "Analyze purchasing trends by year and state"
- "Identify high-risk assets for replacement"

### **AI-Powered Reports**

- **Executive summaries** for senior management
- **Operational insights** for day-to-day management
- **Strategic recommendations** for long-term planning
- **Risk assessments** with mitigation strategies

## 📋 **Professional Reporting**

### **Malaysian Government Format (KEW.PS-6)**

- **Senarai Stok Bertarikh Luput** (Expired Stock List)
- **Official government template** compliance
- **Bilingual support** (Bahasa Malaysia)
- **Automated table generation** with proper formatting

### **Export Formats**

- 📄 **Microsoft Word** (.docx) with embedded charts
- 📄 **PDF** for secure distribution
- 📄 **Plain Text** for further editing
- 📊 **CSV/Excel** for data analysis

## 🎨 **User Interface Features**

### **Professional Styling**

- **Royal Malaysia Police branding** with official colors
- **Blue gradient theme** (#1e40af to #1e3a8a)
- **Responsive design** for desktop and mobile
- **Consistent typography** with professional fonts

### **Interactive Elements**

- **Real-time filtering** by state, category, year
- **Interactive charts** with zoom and pan capabilities
- **Drill-down analysis** from overview to details
- **Export functionality** for all visualizations

### **Accessibility Features**

- **High contrast** text for readability
- **Color-blind friendly** chart palettes
- **Responsive layouts** for different screen sizes
- **Keyboard navigation** support

## 🚀 **Installation & Setup**

### **Local Development**

```bash
# 1. Clone the repository
git clone https://github.com/RajKhalifaa/PDRM.git
cd PDRM

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export OPENAI_API_KEY="your-openai-api-key"

# 5. Run the dashboard
streamlit run streamlit_dashboard.py
```

### **Docker Development**

```bash
# 1. Build the image
docker build -t pdrm-dashboard .

# 2. Run the container
docker run -p 8501:8501 -e OPENAI_API_KEY="your-key" pdrm-dashboard

# 3. Access the dashboard
open http://localhost:8501
```

### **Production Deployment**

```bash
# 1. Pull from Docker Hub
docker pull xxxraj/pdrm-asset-dashboard:latest

# 2. Run with production settings
docker run -d \
  --name pdrm-dashboard \
  --restart unless-stopped \
  -p 8501:8501 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  xxxraj/pdrm-asset-dashboard:latest

# 3. Verify deployment
curl http://localhost:8501/_stcore/health
```

## 🔧 **Configuration**

### **Environment Variables**

| Variable                   | Description                    | Required | Default |
| -------------------------- | ------------------------------ | -------- | ------- |
| `OPENAI_API_KEY`           | OpenAI API key for AI features | Yes      | None    |
| `STREAMLIT_SERVER_PORT`    | Dashboard port                 | No       | 8501    |
| `STREAMLIT_SERVER_ADDRESS` | Bind address                   | No       | 0.0.0.0 |

### **Dashboard Settings**

```python
# streamlit_dashboard.py configuration
PAGE_TITLE = "Jabatan Logistik & Teknologi(JLogT)"
PAGE_ICON = "🏛️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
```

## 📈 **Performance & Monitoring**

### **System Requirements**

- **Memory**: 2GB RAM minimum, 4GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Storage**: 5GB available space
- **Network**: Stable internet for AI features

### **Performance Metrics**

- **Load Time**: < 5 seconds for initial dashboard load
- **Chart Rendering**: < 2 seconds for interactive charts
- **AI Response**: 5-15 seconds for complex queries
- **Report Generation**: 10-30 seconds for full reports

### **Health Monitoring**

- **Health Check Endpoint**: `/_stcore/health`
- **Docker Health Check**: Built-in container monitoring
- **Application Logs**: Comprehensive error tracking
- **Performance Metrics**: Built-in Streamlit metrics

## 🔐 **Security Considerations**

### **Data Protection**

- **No sensitive data** stored in containers
- **Environment variables** for API keys
- **Secure communication** over HTTPS in production
- **Regular security updates** for dependencies

### **API Security**

- **OpenAI API key** protection via environment variables
- **Rate limiting** for AI requests
- **Input validation** for all user inputs
- **Error handling** without sensitive information exposure

## 🚀 **Future Roadmap**

### **Short Term (Q4 2025)**

- [ ] **Real-time notifications** for critical asset status
- [ ] **Mobile app** for field officers
- [ ] **Advanced analytics** with time series forecasting
- [ ] **Integration APIs** for third-party systems

### **Medium Term (Q1-Q2 2026)**

- [ ] **IoT sensor integration** for real-time monitoring
- [ ] **Blockchain** for asset authenticity verification
- [ ] **Advanced ML models** with deep learning
- [ ] **Multi-language support** (Malay, English, Chinese)

### **Long Term (2026+)**

- [ ] **Predictive maintenance** with IoT integration
- [ ] **AR/VR interfaces** for asset inspection
- [ ] **Federal integration** with other government systems
- [ ] **AI-powered recommendations** for procurement

## 🤝 **Contributing**

We welcome contributions from the community! Please see our contributing guidelines:

### **How to Contribute**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Guidelines**

- Follow **PEP 8** style guidelines
- Add **comprehensive tests** for new features
- Update **documentation** for any changes
- Ensure **Docker compatibility** for all changes

## 📞 **Support & Contact**

### **Technical Support**

- **Email**: asset.management@pdrm.gov.my
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/RajKhalifaa/PDRM/issues)
- **Documentation**: Comprehensive guides in `/docs` folder

### **Project Maintainers**

- **Lead Developer**: RajKhalifaa
- **ML Engineer**: Credence Analytics Team
- **DevOps**: Container Solutions Team

## 📄 **License & Legal**

### **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### **Disclaimer**

This is a demonstration system with synthetic data. For production use with real PDRM data, ensure compliance with:

- **Personal Data Protection Act 2010 (PDPA)**
- **Official Secrets Act 1972**
- **PDRM data governance policies**

### **Third-Party Libraries**

- **Streamlit**: Apache License 2.0
- **Plotly**: MIT License
- **OpenAI**: OpenAI API Terms of Service
- **Docker**: Apache License 2.0

---

## 🎯 **Quick Links**

| Resource             | Link                                                                                |
| -------------------- | ----------------------------------------------------------------------------------- |
| 🐳 **Docker Hub**    | [xxxraj/pdrm-asset-dashboard](https://hub.docker.com/r/xxxraj/pdrm-asset-dashboard) |
| 📊 **Live Demo**     | [Dashboard Preview](http://your-demo-url.com)                                       |
| 📚 **Documentation** | [User Guide](DASHBOARD_GUIDE.md)                                                    |
| 🚀 **Deployment**    | [ECS Guide](DEPLOYMENT.md)                                                          |
| 🔄 **Updates**       | [Enhancement Summary](ENHANCEMENT_SUMMARY.md)                                       |

---

**Built with ❤️ for the Royal Malaysia Police**  
**Version**: 1.2.0 | **Last Updated**: September 2025  
**Powered by**: Streamlit • Docker • OpenAI • AWS ECS
