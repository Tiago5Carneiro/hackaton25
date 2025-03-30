/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'placehold.co',
      },
      {
        protocol: 'https',
        hostname: 'flagpedia.net',
      }
    ,
      {
        protocol: 'https',
        hostname: 'media.formula1.com',
    },
        {
            protocol: 'https',
            hostname: 'cdn-1.motorsport.com',
        },
        {
            protocol: 'https',
            hostname: 'picsum.photos',
        },

    ],
  },
  reactStrictMode: true,
}

export default nextConfig;