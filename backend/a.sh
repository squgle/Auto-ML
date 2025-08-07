#!/bin/bash

# Set working directory
PROJECT_ROOT="/home/saurabh/Desktop/Auto-ml/AutoML-Agentic-System/backend"
PROJECT_NAME="django_api"
APPS=("users" "ingestion" "preprocessing" "modeling" "reporting")

echo "🚀 Initializing Django project inside: $PROJECT_ROOT"

cd $PROJECT_ROOT || { echo "❌ Directory not found: $PROJECT_ROOT"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv backend_venv
source backend_venv/bin/activate

# Create requirements.txt
echo "📄 Writing requirements.txt..."
cat <<EOF > requirements.txt
Django>=4.0
djangorestframework
pandas
scikit-learn
matplotlib
reportlab
EOF

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Start Django project
if [ ! -d "$PROJECT_NAME" ]; then
    echo "🛠️ Starting new Django project..."
    django-admin startproject $PROJECT_NAME .
fi

# Create Django apps
for app in "${APPS[@]}"; do
    if [ ! -d "$app" ]; then
        echo "📁 Creating app: $app"
        python manage.py startapp $app
    fi
done

# Add apps to settings.py
echo "⚙️ Updating INSTALLED_APPS in settings.py..."
SETTINGS_FILE="$PROJECT_NAME/settings.py"

for app in "${APPS[@]}"; do
    grep -q "'$app'," $SETTINGS_FILE || \
    sed -i "/'django.contrib.staticfiles',/a\    '$app'," $SETTINGS_FILE
done

grep -q "'rest_framework'," $SETTINGS_FILE || \
sed -i "/'django.contrib.staticfiles',/a\    'rest_framework'," $SETTINGS_FILE

# Add app URL includes to urls.py
echo "🔗 Adding app routes to urls.py..."
URLS_FILE="$PROJECT_NAME/urls.py"

for app in "${APPS[@]}"; do
    grep -q "$app/" $URLS_FILE || \
    sed -i "/urlpatterns = \[/a\    path('api/$app/', include('$app.urls'))," $URLS_FILE
done

# Import include if not present
grep -q "from django.urls import path, include" $URLS_FILE || \
sed -i "1s|from django.urls import path|from django.urls import path, include|" $URLS_FILE

# Create placeholder urls.py for each app
for app in "${APPS[@]}"; do
    echo "📝 Creating basic $app/urls.py"
    mkdir -p "$app"
    cat <<EOF > $app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Add your endpoints here, e.g.:
    # path('example/', views.example_view),
]
EOF
done

# Make migrations and migrate
echo "📦 Running migrations..."
python manage.py makemigrations
python manage.py migrate

# Create superuser prompt
echo "👤 If you'd like to create a superuser, run:"
echo "python manage.py createsuperuser"

# Run server
echo "🚀 Starting Django development server..."
python manage.py runserver

