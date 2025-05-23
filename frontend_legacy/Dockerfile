FROM nginx:alpine

# Copy nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy only the necessary frontend files
COPY frontend/index.html /usr/share/nginx/html/
COPY frontend/styles.css /usr/share/nginx/html/
COPY frontend/app.js /usr/share/nginx/html/
COPY frontend/assets /usr/share/nginx/html/assets/

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"] 