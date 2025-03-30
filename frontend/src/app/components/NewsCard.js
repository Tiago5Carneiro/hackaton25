import Link from 'next/link';
import Image from 'next/image';

export default function NewsCard({ news }) {
  return (
    <div className="card news-card">
      <div className="card-image-container">
        {news.tag && (
          <span className="category-badge">{news.tag}</span>
        )}
        <Image
          src={news.image || "https://placehold.co/600x400"}
          alt={news.title}
          fill
          className="card-image"
        />
      </div>
      
      <div className="card-content">
        <h3 className="card-title">{news.title}</h3>
        <p className="card-description">{news.description}</p>
      </div>
      
      <div className="card-footer">
        <span className="card-date">{news.date || "March 28, 2025"}</span>
        <Link href={`/news/${news.id}`} className="card-link">
          Read more
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" width="16" height="16">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      </div>
    </div>
  );
}