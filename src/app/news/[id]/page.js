"use client";
import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import '../../styles/newsDetail.css';
import Head from 'next/head';

export default function NewsDetailPage() {
  const params = useParams();
  const [article, setArticle] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Sample lorem ipsum content
  const loremContent = `
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam a metus quis lorem malesuada luctus. 
    Cras lacinia, eros at dapibus molestie, risus tortor pretium ligula, eu malesuada tortor eros dapibus mi. 
    Proin laoreet efficitur suscipit. Donec molestie volutpat euismod. Nulla gravida ligula in eros facilisis, 
    sed dignissim tellus aliquam.
    
    Phasellus ac magna vitae neque fringilla finibus. Aenean egestas gravida urna id maximus. Quisque non ligula massa.
    Praesent imperdiet lacus sit amet metus consequat, a vehicula ante sagittis. Pellentesque suscipit a enim sed sodales.
    Curabitur vehicula molestie leo a lobortis. Fusce molestie finibus imperdiet. Donec cursus a ante vel ornare.
    
    Sed fringilla libero id lectus molestie, eu efficitur dui viverra. Cras id orci pulvinar, eleifend velit egestas, 
    convallis ex. Suspendisse vel purus aliquet, maximus justo et, lacinia justo. Nunc posuere ultricies volutpat. 
    Vestibulum dapibus purus at elit auctor, ut dapibus dolor tempor. Cras sagittis enim sit amet nibh porttitor porttitor.
  `;

  useEffect(() => {
    // In a real app, you would fetch this data from an API
    // For now, we'll use the hardcoded news articles from the main page
    const newsArticles = [
      {
        id: 1,
        title: "Verstappen Clinches Third Consecutive Championship After Dramatic Race",
        excerpt: "Max Verstappen secured his third Formula 1 World Championship with a stunning drive at Interlagos, overcoming early setbacks to finish on the podium.",
        category: "races",
        image: "https://picsum.photos/1000/600?random=1",
        date: "March 24, 2025",
        featured: true,
        authorName: "James Allen",
        authorAvatar: "https://picsum.photos/200/200?random=10",
        readTime: "5 min read",
        fullContent: loremContent
      },
      {
        id: 2,
        title: "McLaren Unveils Revolutionary Car Design for 2025",
        excerpt: "McLaren has revealed significant design changes to their 2025 challenger, aiming to challenge Red Bull's dominance with innovative aerodynamic concepts.",
        category: "technical",
        image: "https://picsum.photos/1000/600?random=2",
        date: "March 22, 2025",
        authorName: "Sam Cooper",
        authorAvatar: "https://picsum.photos/200/200?random=11",
        readTime: "4 min read",
        fullContent: loremContent
      },
      {
        id: 3,
        title: "Hamilton Signs Contract Extension with Mercedes",
        excerpt: "Seven-time world champion Lewis Hamilton has committed his future to Mercedes, signing a new multi-year deal that will keep him with the team through 2026.",
        category: "drivers",
        image: "https://picsum.photos/1000/600?random=3",
        date: "March 20, 2025",
        authorName: "Rachel Brooks",
        authorAvatar: "https://picsum.photos/200/200?random=12",
        readTime: "3 min read",
        fullContent: loremContent
      },
      {
        id: 4,
        title: "F1 Announces New Race in Africa for 2026 Season",
        excerpt: "Formula 1 has confirmed that South Africa will return to the calendar in 2026, with Kyalami Circuit set to host the first African Grand Prix since 1993.",
        category: "races",
        image: "https://picsum.photos/1000/600?random=4",
        date: "March 18, 2025",
        authorName: "Lawrence Barretto",
        authorAvatar: "https://picsum.photos/200/200?random=13",
        readTime: "4 min read",
        fullContent: loremContent
      },
      {
        id: 5,
        title: "Ferrari Confirms Major Technical Restructuring",
        excerpt: "Scuderia Ferrari has announced a significant reshuffle of its technical department following a challenging start to the 2025 season.",
        category: "teams",
        image: "https://picsum.photos/1000/600?random=5",
        date: "March 16, 2025",
        authorName: "Anna Sanchez",
        authorAvatar: "https://picsum.photos/200/200?random=14",
        readTime: "5 min read",
        fullContent: loremContent
      },
      {
        id: 6,
        title: "New Sustainable Fuel Regulations Set for 2026",
        excerpt: "The FIA has finalized the new sustainable fuel regulations that will come into effect from the 2026 season, targeting a significant reduction in carbon emissions.",
        category: "technical",
        image: "https://picsum.photos/1000/600?random=6",
        date: "March 14, 2025",
        authorName: "Mark Hughes",
        authorAvatar: "https://picsum.photos/200/200?random=15",
        readTime: "6 min read",
        fullContent: loremContent
      },
      {
        id: 7,
        title: "Piastri Secures Maiden Victory in Thrilling Italian Grand Prix",
        excerpt: "Oscar Piastri claimed his first Formula 1 win at Monza after a race-long battle with teammate Lando Norris and Charles Leclerc's Ferrari.",
        category: "races",
        image: "https://picsum.photos/1000/600?random=7",
        date: "March 12, 2025",
        authorName: "Will Buxton",
        authorAvatar: "https://picsum.photos/200/200?random=16",
        readTime: "5 min read",
        fullContent: loremContent
      },
      {
        id: 8,
        title: "Alonso Announces Retirement at End of 2025 Season",
        excerpt: "Two-time world champion Fernando Alonso has announced that the 2025 season will be his last in Formula 1, bringing an end to a legendary career spanning over two decades.",
        category: "drivers",
        image: "https://picsum.photos/1000/600?random=8",
        date: "March 10, 2025",
        authorName: "Natalie Pinkham",
        authorAvatar: "https://picsum.photos/200/200?random=17",
        readTime: "7 min read",
        fullContent: loremContent
      },
      {
        id: 9,
        title: "Red Bull Debuts New Innovative DRS System",
        excerpt: "Red Bull Racing has introduced a revolutionary rear wing design that maximizes DRS effectiveness, pending FIA approval for its legality.",
        category: "technical",
        image: "https://picsum.photos/1000/600?random=9",
        date: "March 8, 2025",
        authorName: "Ted Kravitz",
        authorAvatar: "https://picsum.photos/200/200?random=18",
        readTime: "5 min read",
        fullContent: loremContent
      }
    ];

    if (params.id) {
      const fetchedArticle = newsArticles.find(item => item.id === parseInt(params.id));
      setArticle(fetchedArticle);
      
      // Add a small delay for animations
      setTimeout(() => {
        setIsLoaded(true);
      }, 200);
    }
  }, [params.id]);

  if (!article) {
    return (
      <div className="loading-container">
        <div className="loader"></div>
        <p>Loading article...</p>
      </div>
    );
  }

  return (
    <>
      <Head>
        <link 
          rel="stylesheet" 
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" 
          integrity="sha512-1ycn6IcaQQ40/MKBW2W4Rhis/DbILU74C1vSrLJxCq57o941Ym01SwNsOMqvEBFlcgUa6xLiPY/NS5R+E6ztJQ==" 
          crossOrigin="anonymous" 
          referrerPolicy="no-referrer" 
        />
      </Head>
      <div className={`news-detail-container ${isLoaded ? 'loaded' : ''}`}>
        <Link href="/news" className="back-link">
          ← Back to News
        </Link>
        
        <h1 className="news-detail-title">{article.title}</h1>
        
        <div className="news-detail-meta">
          <span className="news-detail-date">{article.date}</span>
          <span className="news-detail-category">{article.category}</span>
          <span className="news-detail-readtime">{article.readTime}</span>
        </div>
        
        <div className="news-detail-image-container">
          <Image
            src={article.image}
            alt={article.title}
            width={800}
            height={450}
            layout="responsive"
            className="news-detail-image"
            priority
          />
        </div>
        
        <div className="news-detail-author-info">
          <div className="author-avatar-container">
            <Image
              src={article.authorAvatar}
              alt={article.authorName}
              width={60}
              height={60}
              className="author-avatar"
            />
          </div>
          <div className="author-details">
            <span className="author-name">{article.authorName}</span>
            <span className="author-role">F1 Correspondent</span>
          </div>
        </div>
        
        <p className="news-detail-excerpt">{article.excerpt}</p>
        
        <div className="news-detail-content">
          {article.fullContent.split('\n\n').map((paragraph, index) => (
            <p key={index}>{paragraph}</p>
          ))}
        </div>
        
        <div className="news-detail-footer">
          <div className="news-detail-tags">
            <span className="tag">Formula 1</span>
            <span className="tag">{article.category}</span>
            <span className="tag">2025 Season</span>
          </div>
          
          <div className="news-detail-share">
            <span>Share this article:</span>
            <div className="social-icons">
              <a href="#" aria-label="Twitter">
                <svg className="social-icon" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                </svg>
              </a>
              <a href="#" aria-label="Instagram">
                <svg className="social-icon" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path fillRule="evenodd" d="M12.315 2c2.43 0 2.784.013 3.808.06 1.064.049 1.791.218 2.427.465a4.902 4.902 0 011.772 1.153 4.902 4.902 0 011.153 1.772c.247.636.416 1.363.465 2.427.048 1.067.06 1.407.06 4.123v.08c0 2.643-.012 2.987-.06 4.043-.049 1.064-.218 1.791-.465 2.427a4.902 4.902 0 01-1.153 1.772 4.902 4.902 0 01-1.772 1.153c-.636.247-1.363.416-2.427.465-1.067.048-1.407.06-4.123.06h-.08c-2.643 0-2.987-.012-4.043-.06-1.064-.049-1.791-.218-2.427-.465a4.902 4.902 0 01-1.772-1.153 4.902 4.902 0 01-1.153-1.772c-.247-.636-.416-1.363-.465-2.427-.047-1.024-.06-1.379-.06-3.808v-.63c0-2.43.013-2.784.06-3.808.049-1.064.218-1.791.465-2.427a4.902 4.902 0 011.153-1.772A4.902 4.902 0 015.45 2.525c.636-.247 1.363-.416 2.427-.465C8.901 2.013 9.256 2 11.685 2h.63zm-.081 1.802h-.468c-2.456 0-2.784.011-3.807.058-.975.045-1.504.207-1.857.344-.467.182-.8.398-1.15.748-.35.35-.566.683-.748 1.15-.137.353-.3.882-.344 1.857-.047 1.023-.058 1.351-.058 3.807v.468c0 2.456.011 2.784.058 3.807.045.975.207 1.504.344 1.857.182.466.399.8.748 1.15.35.35.683.566 1.15.748.353.137.882.3 1.857.344 1.054.048 1.37.058 4.041.058h.08c2.597 0 2.917-.01 3.96-.058.976-.045 1.505-.207 1.858-.344.466-.182.8-.398 1.15-.748.35-.35.566-.683.748-1.15.137-.353.3-.882.344-1.857.048-1.055.058-1.37.058-4.041v-.08c0-2.597-.01-2.917-.058-3.96-.045-.976-.207-1.505-.344-1.858a3.097 3.097 0 00-.748-1.15 3.098 3.098 0 00-1.15-.748c-.353-.137-.882-.3-1.857-.344-1.023-.047-1.351-.058-3.807-.058zM12 6.865a5.135 5.135 0 110 10.27 5.135 5.135 0 010-10.27zm0 1.802a3.333 3.333 0 100 6.666 3.333 3.333 0 000-6.666zm5.338-3.205a1.2 1.2 0 110 2.4 1.2 1.2 0 010-2.4z" clipRule="evenodd" />
                </svg>
              </a>
              <a href="#" aria-label="YouTube">
                <svg className="social-icon" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0C1.5 3.42.4 5.5.4 9.98v4.04c0 4.509 1.1 6.62 3.985 6.827 3.6.245 11.626.246 15.23 0C22.5 20.55 23.6 18.5 23.6 14v-4c0-4.484-1.1-6.62-3.985-6.816zM9.6 15.998V8.002L16.8 12l-7.2 3.998z" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}