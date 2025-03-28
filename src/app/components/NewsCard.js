import Link from 'next/link';
import Image from 'next/image';

export default function NewsCard({ news }) {
  return (
    <Link href={`/news/${news.id}`}>
      <div className="bg-white rounded-lg shadow-md overflow-hidden h-full hover:shadow-lg transition-shadow">
        <div className="relative h-48">
          <Image
            src={news.image || "https://via.placeholder.com/400x200?text=F1+News"}
            alt={news.title}
            fill
            style={{ objectFit: 'cover' }}
          />
          {news.tag && (
            <span className="absolute top-2 right-2 bg-red-600 text-white text-xs px-2 py-1 rounded">
              {news.tag}
            </span>
          )}
        </div>
        
        <div className="p-4">
          <h3 className="text-xl font-bold mb-2 line-clamp-2">{news.title}</h3>
          <p className="text-gray-700 mb-4 line-clamp-3">{news.description}</p>
          
          <div className="flex justify-between items-center text-sm">
            <span className="text-gray-500">{news.date || "March 28, 2025"}</span>
            <span className="text-red-600 font-medium">Read more</span>
          </div>
        </div>
      </div>
    </Link>
  );
}