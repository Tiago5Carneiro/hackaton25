"use client";
import { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';

export default function NewsPage() {
  const [activeCategory, setActiveCategory] = useState('all');
  const [isLoaded, setIsLoaded] = useState(false);
  
  useEffect(() => {
    // Pequeno delay para garantir que as animações ocorram após a renderização
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 200);
    
    return () => clearTimeout(timer);
  }, []);
  
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
      image: "https://placehold.co/800x450?text=Verstappen+Champion",
      date: "March 24, 2025",
      featured: true,
      authorName: "James Allen",
      authorAvatar: "https://placehold.co/150x150?text=JA",
      readTime: "5 min read"
    },
    {
      id: 2,
      title: "McLaren Unveils Revolutionary Car Design for 2025",
      excerpt: "McLaren has revealed significant design changes to their 2025 challenger, aiming to challenge Red Bull's dominance with innovative aerodynamic concepts.",
      category: "technical",
      image: "https://placehold.co/800x450?text=McLaren+2025",
      date: "March 22, 2025",
      authorName: "Sam Cooper",
      authorAvatar: "https://placehold.co/150x150?text=SC",
      readTime: "4 min read"
    },
    {
      id: 3,
      title: "Hamilton Signs Contract Extension with Mercedes",
      excerpt: "Seven-time world champion Lewis Hamilton has committed his future to Mercedes, signing a new multi-year deal that will keep him with the team through 2026.",
      category: "drivers",
      image: "https://placehold.co/800x450?text=Hamilton+Contract",
      date: "March 20, 2025",
      authorName: "Rachel Brooks",
      authorAvatar: "https://placehold.co/150x150?text=RB",
      readTime: "3 min read"
    },
    {
      id: 4,
      title: "F1 Announces New Race in Africa for 2026 Season",
      excerpt: "Formula 1 has confirmed that South Africa will return to the calendar in 2026, with Kyalami Circuit set to host the first African Grand Prix since 1993.",
      category: "races",
      image: "https://placehold.co/800x450?text=African+GP",
      date: "March 18, 2025",
      authorName: "Lawrence Barretto",
      authorAvatar: "https://placehold.co/150x150?text=LB",
      readTime: "4 min read"
    },
    {
      id: 5,
      title: "Ferrari Confirms Major Technical Restructuring",
      excerpt: "Scuderia Ferrari has announced a significant reshuffle of its technical department following a challenging start to the 2025 season.",
      category: "teams",
      image: "https://placehold.co/800x450?text=Ferrari+Tech+Team",
      date: "March 16, 2025",
      authorName: "Anna Sanchez",
      authorAvatar: "https://placehold.co/150x150?text=AS",
      readTime: "5 min read"
    },
    {
      id: 6,
      title: "New Sustainable Fuel Regulations Set for 2026",
      excerpt: "The FIA has finalized the new sustainable fuel regulations that will come into effect from the 2026 season, targeting a significant reduction in carbon emissions.",
      category: "technical",
      image: "https://placehold.co/800x450?text=Sustainable+Fuel",
      date: "March 14, 2025",
      authorName: "Mark Hughes",
      authorAvatar: "https://placehold.co/150x150?text=MH",
      readTime: "6 min read"
    },
    {
      id: 7,
      title: "Piastri Secures Maiden Victory in Thrilling Italian Grand Prix",
      excerpt: "Oscar Piastri claimed his first Formula 1 win at Monza after a race-long battle with teammate Lando Norris and Charles Leclerc's Ferrari.",
      category: "races",
      image: "https://placehold.co/800x450?text=Piastri+Win",
      date: "March 12, 2025",
      authorName: "Will Buxton",
      authorAvatar: "https://placehold.co/150x150?text=WB",
      readTime: "5 min read"
    },
    {
      id: 8,
      title: "Alonso Announces Retirement at End of 2025 Season",
      excerpt: "Two-time world champion Fernando Alonso has announced that the 2025 season will be his last in Formula 1, bringing an end to a legendary career spanning over two decades.",
      category: "drivers",
      image: "https://placehold.co/800x450?text=Alonso+Retirement",
      date: "March 10, 2025",
      authorName: "Natalie Pinkham",
      authorAvatar: "https://placehold.co/150x150?text=NP",
      readTime: "7 min read"
    },
    {
      id: 9,
      title: "Red Bull Debuts New Innovative DRS System",
      excerpt: "Red Bull Racing has introduced a revolutionary rear wing design that maximizes DRS effectiveness, pending FIA approval for its legality.",
      category: "technical",
      image: "https://placehold.co/800x450?text=Red+Bull+DRS",
      date: "March 8, 2025",
      authorName: "Ted Kravitz",
      authorAvatar: "https://placehold.co/150x150?text=TK",
      readTime: "5 min read"
    }
  ];
  
  const filteredNews = activeCategory === 'all' 
    ? newsArticles 
    : newsArticles.filter(article => article.category === activeCategory);
  
  const featuredNews = filteredNews.filter(article => article.featured);
  const regularNews = filteredNews.filter(article => !article.featured);

  // Variantes de animação para os cards
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        type: "spring",
        stiffness: 80,
        damping: 12
      }
    }
  };

  const titleVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        type: "spring",
        stiffness: 100,
        damping: 15
      }
    }
  };
  
  return (
    <div className="news-container">
      <motion.h1 
        className="news-title"
        variants={titleVariants}
        initial="hidden"
        animate={isLoaded ? "visible" : "hidden"}
      >
        Latest Formula 1 News
      </motion.h1>
      
      {/* Category filters */}
      <motion.div 
        className="category-filters"
        initial={{ opacity: 0, y: -10 }}
        animate={isLoaded ? { opacity: 1, y: 0 } : { opacity: 0, y: -10 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        {categories.map((category, index) => (
          <motion.button
            key={category.id}
            onClick={() => setActiveCategory(category.id)}
            className={`category-button ${activeCategory === category.id ? 'active' : ''}`}
            initial={{ opacity: 0, y: -10 }}
            animate={isLoaded ? { opacity: 1, y: 0 } : { opacity: 0, y: -10 }}
            transition={{ delay: 0.2 + (index * 0.05), duration: 0.4 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {category.name}
          </motion.button>
        ))}
      </motion.div>
      
      {/* Featured news section */}
      {featuredNews.length > 0 && (
        <section className="featured-section">
          <motion.h2 
            className="section-title"
            initial={{ opacity: 0, x: -20 }}
            animate={isLoaded ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            Featured Stories
          </motion.h2>
          <motion.div 
            className="news-grid featured-grid"
            variants={containerVariants}
            initial="hidden"
            animate={isLoaded ? "visible" : "hidden"}
          >
            {featuredNews.map((article, index) => (
              <motion.div 
                key={article.id}
                variants={itemVariants}
                custom={index}
                whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
              >
                <Link href={`/news/${article.id}`} className={index === 0 ? 'main-feature' : 'news-card'}>
                  <div className="news-image-container">
                    <Image
                      src={article.image}
                      alt={article.title}
                      fill
                      style={{ objectFit: 'cover' }}
                    />
                    <div className="featured-tag">FEATURED</div>
                  </div>
                  
                  <div className="news-content">
                    <div className="news-meta">
                      <span className="news-date">{article.date}</span>
                      <span className="news-category">{article.category}</span>
                    </div>
                    
                    <h2 className="news-headline">{article.title}</h2>
                    <p className="news-excerpt">{article.excerpt}</p>
                    
                    <div className="news-footer">
                      <div className="news-author">
                        <div className="author-avatar">
                          <Image
                            src={article.authorAvatar}
                            alt={article.authorName}
                            fill
                            style={{ objectFit: 'cover' }}
                          />
                        </div>
                        <span className="author-name">{article.authorName}</span>
                      </div>
                      <span className="read-time">{article.readTime}</span>
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}
          </motion.div>
        </section>
      )}
      
      {/* Regular news articles */}
      <section>
        <motion.h2 
          className="section-title"
          initial={{ opacity: 0, x: -20 }}
          animate={isLoaded ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
          transition={{ delay: 0.4, duration: 0.4 }}
        >
          Latest News
        </motion.h2>
        <motion.div 
          className="news-grid regular-grid"
          variants={containerVariants}
          initial="hidden"
          animate={isLoaded ? "visible" : "hidden"}
        >
          {regularNews.map((article, index) => (
            <motion.div 
              key={article.id}
              variants={itemVariants}
              custom={index}
              whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
            >
              <Link href={`/news/${article.id}`} className="news-card">
                <div className="news-image-container">
                  <Image
                    src={article.image}
                    alt={article.title}
                    fill
                    style={{ objectFit: 'cover' }}
                  />
                </div>
                
                <div className="news-content">
                  <div className="news-meta">
                    <span className="news-date">{article.date}</span>
                    <span className="news-category">{article.category}</span>
                  </div>
                  
                  <h2 className="news-headline">{article.title}</h2>
                  <p className="news-excerpt">{article.excerpt}</p>
                  
                  <div className="news-footer">
                    <div className="news-author">
                      <div className="author-avatar">
                        <Image
                          src={article.authorAvatar}
                          alt={article.authorName}
                          fill
                          style={{ objectFit: 'cover' }}
                        />
                      </div>
                      <span className="author-name">{article.authorName}</span>
                    </div>
                    <span className="read-time">{article.readTime}</span>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>
        
        {filteredNews.length === 0 && (
          <motion.div 
            className="empty-state"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <p>No news articles found in this category.</p>
          </motion.div>
        )}
      </section>
    </div>
  );
}