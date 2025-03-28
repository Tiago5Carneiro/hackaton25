"use client";
import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';

export default function NewsPage() {
  const [activeCategory, setActiveCategory] = useState('all');
  
  const categories = [
    { id: 'all', name: 'All News' },
    { id: 'races', name: 'Race Reports' },
    { id: 'drivers', name: 'Drivers' },
    { id: 'teams', name: 'Teams' },
    { id: 'technical', name: 'Technical' },
  ];
  
  const newsArticles = [
    {
      id: 1,
      title: "Verstappen Clinches Third Consecutive Championship After Dramatic Race",
      excerpt: "Max Verstappen secured his third Formula 1 World Championship with a stunning drive at Interlagos, overcoming early setbacks to finish on the podium.",
      category: "races",
      image: "https://via.placeholder.com/800x450?text=Verstappen+Champion",
      date: "March 24, 2025",
      featured: true,
      authorName: "James Allen",
      authorAvatar: "https://via.placeholder.com/50?text=JA",
      readTime: "5 min read"
    },
    {
      id: 2,
      title: "McLaren Unveils Revolutionary Car Design for 2025",
      excerpt: "McLaren has revealed significant design changes to their 2025 challenger, aiming to challenge Red Bull's dominance with innovative aerodynamic concepts.",
      category: "technical",
      image: "https://via.placeholder.com/800x450?text=McLaren+2025",
      date: "March 22, 2025",
      featured: true,
      authorName: "Sam Cooper",
      authorAvatar: "https://via.placeholder.com/50?text=SC",
      readTime: "4 min read"
    },
    {
      id: 3,
      title: "Hamilton Signs Contract Extension with Mercedes",
      excerpt: "Seven-time world champion Lewis Hamilton has committed his future to Mercedes, signing a new multi-year deal that will keep him with the team through 2026.",
      category: "drivers",
      image: "https://via.placeholder.com/800x450?text=Hamilton+Contract",
      date: "March 20, 2025",
      featured: true,
      authorName: "Rachel Brooks",
      authorAvatar: "https://via.placeholder.com/50?text=RB",
      readTime: "3 min read"
    },
    {
      id: 4,
      title: "F1 Announces New Race in Africa for 2026 Season",
      excerpt: "Formula 1 has confirmed that South Africa will return to the calendar in 2026, with Kyalami Circuit set to host the first African Grand Prix since 1993.",
      category: "races",
      image: "https://via.placeholder.com/400x225?text=African+GP",
      date: "March 18, 2025",
      authorName: "Lawrence Barretto",
      authorAvatar: "https://via.placeholder.com/50?text=LB",
      readTime: "4 min read"
    },
    {
      id: 5,
      title: "Ferrari Confirms Major Technical Restructuring",
      excerpt: "Scuderia Ferrari has announced a significant reshuffle of its technical department following a challenging start to the 2025 season.",
      category: "teams",
      image: "https://via.placeholder.com/400x225?text=Ferrari+Changes",
      date: "March 16, 2025",
      authorName: "Anna Sanchez",
      authorAvatar: "https://via.placeholder.com/50?text=AS",
      readTime: "5 min read"
    },
    {
      id: 6,
      title: "New Sustainable Fuel Regulations Set for 2026",
      excerpt: "The FIA has finalized the new sustainable fuel regulations that will come into effect from the 2026 season, targeting a significant reduction in carbon emissions.",
      category: "technical",
      image: "https://via.placeholder.com/400x225?text=Sustainable+Fuel",
      date: "March 14, 2025",
      authorName: "Mark Hughes",
      authorAvatar: "https://via.placeholder.com/50?text=MH",
      readTime: "6 min read"
    },
    {
      id: 7,
      title: "Piastri Secures Maiden Victory in Thrilling Italian Grand Prix",
      excerpt: "Oscar Piastri claimed his first Formula 1 win at Monza after a race-long battle with teammate Lando Norris and Charles Leclerc's Ferrari.",
      category: "races",
      image: "https://via.placeholder.com/400x225?text=Piastri+Win",
      date: "March 12, 2025",
      authorName: "Will Buxton",
      authorAvatar: "https://via.placeholder.com/50?text=WB",
      readTime: "5 min read"
    },
    {
      id: 8,
      title: "Alonso Announces Retirement at End of 2025 Season",
      excerpt: "Two-time world champion Fernando Alonso has announced that the 2025 season will be his last in Formula 1, bringing an end to a legendary career spanning over two decades.",
      category: "drivers",
      image: "https://via.placeholder.com/400x225?text=Alonso+Retirement",
      date: "March 10, 2025",
      authorName: "Natalie Pinkham",
      authorAvatar: "https://via.placeholder.com/50?text=NP",
      readTime: "7 min read"
    },
    {
      id: 9,
      title: "Red Bull Debuts New Innovative DRS System",
      excerpt: "Red Bull Racing has introduced a revolutionary rear wing design that maximizes DRS effectiveness, pending FIA approval for its legality.",
      category: "technical",
      image: "https://via.placeholder.com/400x225?text=Red+Bull+DRS",
      date: "March 8, 2025",
      authorName: "Ted Kravitz",
      authorAvatar: "https://via.placeholder.com/50?text=TK",
      readTime: "5 min read"
    }
  ];
  
  const filteredNews = activeCategory === 'all' 
    ? newsArticles 
    : newsArticles.filter(article => article.category === activeCategory);
  
  const featuredNews = filteredNews.filter(article => article.featured);
  const regularNews = filteredNews.filter(article => !article.featured);
  
  return (
    <div className="container mx-auto">
      <h1 className="text-3xl font-bold mb-8">Latest Formula 1 News</h1>
      
      {/* Category filters */}
      <div className="flex flex-wrap gap-2 mb-8">
        {categories.map(category => (
          <button
            key={category.id}
            onClick={() => setActiveCategory(category.id)}
            className={`px-4 py-2 rounded-full text-sm font-medium ${
              activeCategory === category.id
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
            }`}
          >
            {category.name}
          </button>
        ))}
      </div>
      
      {/* Featured news section */}
      {featuredNews.length > 0 && (
        <section className="mb-12">
          <div className="grid md:grid-cols-3 gap-6">
            {featuredNews.map((article, index) => (
              <Link key={article.id} href={`/news/${article.id}`}>
                <div className={`bg-white rounded-lg shadow-md overflow-hidden h-full hover:shadow-lg transition-shadow ${
                  index === 0 ? 'md:col-span-3 lg:grid lg:grid-cols-2 lg:gap-6' : ''
                }`}>
                  <div className={`relative ${index === 0 ? 'h-64 lg:h-auto' : 'h-48'}`}>
                    <Image
                      src={article.image}
                      alt={article.title}
                      fill
                      style={{ objectFit: 'cover' }}
                    />
                    <div className="absolute top-0 left-0 bg-red-600 text-white text-xs px-2 py-1">
                      FEATURED
                    </div>
                  </div>
                  
                  <div className="p-4">
                    <div className="flex items-center mb-2">
                      <span className="text-xs text-gray-500 mr-2">{article.date}</span>
                      <span className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded capitalize">{article.category}</span>
                    </div>
                    
                    <h2 className={`font-bold mb-2 ${index === 0 ? 'text-2xl' : 'text-xl'}`}>{article.title}</h2>
                    <p className="text-gray-600 mb-4">{article.excerpt}</p>
                    
                    <div className="flex justify-between items-center">
                      <div className="flex items-center">
                        <div className="w-6 h-6 rounded-full overflow-hidden mr-2">
                          <Image
                            src={article.authorAvatar}
                            alt={article.authorName}
                            width={24}
                            height={24}
                          />
                        </div>
                        <span className="text-xs">{article.authorName}</span>
                      </div>
                      <span className="text-xs text-gray-500">{article.readTime}</span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}
      
      {/* Regular news articles */}
      <section>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {regularNews.map(article => (
            <Link key={article.id} href={`/news/${article.id}`}>
              <div className="bg-white rounded-lg shadow-md overflow-hidden h-full hover:shadow-lg transition-shadow">
                <div className="relative h-48">
                  <Image
                    src={article.image}
                    alt={article.title}
                    fill
                    style={{ objectFit: 'cover' }}
                  />
                </div>
                
                <div className="p-4">
                  <div className="flex items-center mb-2">
                    <span className="text-xs text-gray-500 mr-2">{article.date}</span>
                    <span className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded capitalize">{article.category}</span>
                  </div>
                  
                  <h2 className="text-xl font-bold mb-2">{article.title}</h2>
                  <p className="text-gray-600 mb-4 line-clamp-2">{article.excerpt}</p>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <div className="w-6 h-6 rounded-full overflow-hidden mr-2">
                        <Image
                          src={article.authorAvatar}
                          alt={article.authorName}
                          width={24}
                          height={24}
                        />
                      </div>
                      <span className="text-xs">{article.authorName}</span>
                    </div>
                    <span className="text-xs text-gray-500">{article.readTime}</span>
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>
        
        {filteredNews.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-600">No news articles found in this category.</p>
          </div>
        )}
      </section>
    </div>
  );
}