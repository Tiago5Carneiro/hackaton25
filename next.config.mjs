/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
      remotePatterns: [
        {
          protocol: 'https',
          hostname: 'via.placeholder.com',
          pathname: '/**',
        },
        // Add any other domains you might need, like:
        // {
        //   protocol: 'https',
        //   hostname: 'example.com',
        //   pathname: '/**',
        // }
      ],
    },
  }
  
  export default nextConfig;