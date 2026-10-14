FROM nginx:1.27-alpine

# replace the default config with ours (it sends all requests to the Flask app)
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
